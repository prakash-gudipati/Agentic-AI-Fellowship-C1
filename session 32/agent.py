"""
Session 32 — agent.py

The minimal memory-aware agent loop.

This is intentionally NOT a full ReAct agent. S29 covered that loop end
to end. Session 32 is about MEMORY and CONTEXT ENGINEERING, so the
agent loop here is the simplest thing that exercises every memory layer:

    user turn
      -> append to WORKING_MEMORY
      -> WORKING_MEMORY may EVICT old turns
      -> evicted turns flow into SUMMARISER (PROD PATTERN: Summarisation)
      -> evicted turns also flow into SEMANTIC_MEMORY (PROD PATTERN: Recall)
      -> CONTEXT_BUILDER assembles {system, summary, semantic hits, working, user}
      -> LLM produces an assistant turn
      -> assistant turn appended to WORKING_MEMORY (same eviction logic)
      -> repeat

At session end:
  -> build_snapshot() collects state from every layer
  -> SessionStore.save() writes the snapshot to disk
                                       (PROD PATTERN: Session State as Artifact)

PRODUCTION SUBSTRATE (optional):
  Pass store=MemoryStore(...) to the constructor. Every state change is
  ALSO written to the SQLite-backed store. Use MemoryAgent.from_store(
  store, session_id) to rehydrate a fresh agent from the DB. The
  in-memory path is unchanged when store is None — same code, same demos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from context_builder import BuiltContext, ContextBuilder
from eviction import EvictionPolicy, HybridPolicy
from llm_client import LLMClient
from memory_store import MemoryStore
from memory_types import (
    EvictionEvent,
    SemanticHit,
    SessionStateSnapshot,
    SummaryNote,
    Turn,
)
from semantic_memory import SemanticMemory
from session_state import build_snapshot, new_session_id
from summariser import Summariser
from working_memory import WorkingMemory


# ----------------------------------------------------------------------------
# Trace returned per turn so the walkthrough can show every layer
# ----------------------------------------------------------------------------


@dataclass
class TurnTrace:
    user_text: str
    assistant_text: str
    evicted: List[Turn] = field(default_factory=list)
    eviction_events: List[EvictionEvent] = field(default_factory=list)
    semantic_hits: List[SemanticHit] = field(default_factory=list)
    summary_after: Optional[SummaryNote] = None
    context: Optional[BuiltContext] = None
    summarised: bool = False


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------


AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant inside an Agentic AI memory demo. "
    "You will be given a SUMMARY of earlier conversation, possibly some "
    "RELEVANT PRIOR TURNS retrieved from semantic memory, and the recent "
    "VERBATIM turns. Use all of that context to answer the user's latest "
    "message accurately and briefly. If a fact is in memory, recall it "
    "directly — do not say you cannot remember. If a fact is genuinely "
    "absent from every memory layer, say so plainly."
)


class MemoryAgent:
    """
    The integration point for the four PROD PATTERNS.

    Constructor switches let the demos toggle layers on and off:
      - enable_summary  : turn the rolling Summariser on/off
      - enable_semantic : turn vector recall on/off
      - working_budget  : token budget on the verbatim buffer
      - working_policy  : eviction policy implementation
      - store           : optional MemoryStore for durable persistence

    When store is provided, every memory mutation is mirrored to the
    SQLite-backed store. The in-memory layers remain the agent's hot
    path; the store is the durable shadow.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        working_budget: int = 800,
        working_policy: Optional[EvictionPolicy] = None,
        enable_summary: bool = True,
        enable_semantic: bool = True,
        semantic_k: int = 3,
        semantic_min_score: float = 0.15,
        session_id: Optional[str] = None,
        store: Optional[MemoryStore] = None,
    ) -> None:
        self.client = client or LLMClient()
        self.session_id = session_id or new_session_id()
        self.working = WorkingMemory(
            token_budget=working_budget,
            policy=working_policy or HybridPolicy(k=4, min_keep=2),
        )
        self.summariser = Summariser(client=self.client)
        self.semantic = SemanticMemory()
        self.builder = ContextBuilder()
        self.summary: Optional[SummaryNote] = None
        self.eviction_log: List[EvictionEvent] = []
        self.enable_summary = enable_summary
        self.enable_semantic = enable_semantic
        self.semantic_k = semantic_k
        self.semantic_min_score = semantic_min_score
        self.store = store
        if self.store is not None:
            self.store.upsert_session(
                self.session_id,
                metadata={
                    "summary_enabled": enable_summary,
                    "semantic_enabled": enable_semantic,
                },
            )

    # ------------------------------------------------------------------
    # Chat — one user message -> one assistant reply
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> TurnTrace:
        trace = TurnTrace(user_text=user_message, assistant_text="")

        # 1) Semantic recall BEFORE we mutate working memory.
        hits: List[SemanticHit] = []
        if self.enable_semantic:
            hits = self._semantic_recall(user_message)
        trace.semantic_hits = hits

        # 2) Assemble the prompt.
        context = self.builder.build(
            system_text=AGENT_SYSTEM_PROMPT,
            summary=self.summary,
            semantic_hits=hits,
            working_turns=self.working.turns(),
            user_message=user_message,
        )
        trace.context = context

        # 3) Call the model.
        reply = self.client.complete_turn(
            system=context.system,
            messages=context.messages,
        )
        if not reply:
            reply = "(empty reply)"
        trace.assistant_text = reply

        # 4) Append the new turns to working memory and run eviction.
        user_turn = Turn(role="user", content=user_message)
        self._append_with_memory_side_effects(user_turn, trace)

        assistant_turn = Turn(role="assistant", content=reply)
        self._append_with_memory_side_effects(assistant_turn, trace)

        trace.summary_after = self.summary
        return trace

    # ------------------------------------------------------------------
    # Snapshot — at session end (JSON-artifact path; still works alongside store)
    # ------------------------------------------------------------------

    def snapshot(self) -> SessionStateSnapshot:
        return build_snapshot(
            session_id=self.session_id,
            working_turns=self.working.snapshot(),
            summary_notes=[self.summary] if self.summary else [],
            semantic_turn_ids=self.semantic.turn_ids(),
            eviction_log=list(self.eviction_log),
            metadata={
                "ended_at": time.time(),
                "summary_enabled": self.enable_summary,
                "semantic_enabled": self.enable_semantic,
                "store_path": getattr(self.store, "db_path", None),
            },
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SessionStateSnapshot,
        client: Optional[LLMClient] = None,
    ) -> "MemoryAgent":
        agent = cls(client=client, session_id=snapshot.session_id)
        for t in snapshot.working_turns:
            agent.working.append(t)
        if snapshot.summary_notes:
            agent.summary = snapshot.summary_notes[-1]
        all_known = {t.turn_id: t for t in snapshot.working_turns}
        for t_id in snapshot.semantic_turn_ids:
            t = all_known.get(t_id)
            if t is not None:
                agent.semantic.add(t)
        agent.eviction_log = list(snapshot.eviction_log)
        return agent

    # ------------------------------------------------------------------
    # NEW — Rehydrate from a MemoryStore (the DB path)
    # ------------------------------------------------------------------

    @classmethod
    def from_store(
        cls,
        store: MemoryStore,
        session_id: str,
        client: Optional[LLMClient] = None,
        **kwargs,
    ) -> "MemoryAgent":
        """
        Build a fresh MemoryAgent populated from a SQLite store.

        Every memory layer is reconstructed by querying the DB:
          - working memory   : turns table where in_working = 1
          - rolling summary  : summaries table
          - semantic memory  : vectors table (loaded into the in-memory
                               SemanticMemory so the hot-path recall API
                               is unchanged)
          - eviction log     : eviction_log table

        The store stays attached, so subsequent .chat() calls continue
        to write through to the same DB.
        """

        agent = cls(client=client, session_id=session_id, store=store, **kwargs)

        # Working memory — restore in chronological order.
        for t in store.working_turns(session_id):
            agent.working.append(t)

        # Summary
        agent.summary = store.get_summary(session_id)

        # Semantic memory — rebuild in-memory layer from (turn, vector) pairs.
        for turn, vector in store.session_vectors(session_id):
            # SemanticMemory expects to embed itself. We bypass that by
            # directly appending to its private state so the vector loaded
            # from disk is the EXACT vector we computed at eviction time.
            agent.semantic._turns.append(turn)
            agent.semantic._vectors.append(vector)

        # Eviction log
        agent.eviction_log = list(store.eviction_log(session_id))

        return agent

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _semantic_recall(self, query: str) -> List[SemanticHit]:
        """
        Semantic recall — DB path if a store is attached, in-process otherwise.

        Both paths use the same scoring math (cosine over normalised vectors).
        The DB path is more honest about production: vectors live on disk,
        the query loads candidates, similarity is computed on top.
        """

        if self.store is not None:
            q_vec = self.semantic.embedder.embed(query)
            return self.store.recall(
                self.session_id,
                q_vec,
                k=self.semantic_k,
                min_score=self.semantic_min_score,
            )
        return self.semantic.recall(
            query,
            k=self.semantic_k,
            min_score=self.semantic_min_score,
        )

    def _append_with_memory_side_effects(
        self, turn: Turn, trace: TurnTrace
    ) -> None:
        result = self.working.append(turn)

        # Mirror the append into the DB (in_working=1) BEFORE the eviction
        # we just triggered. This way the eviction-log row always points
        # at a turn that the DB already knows about.
        if self.store is not None:
            self.store.append_turn(self.session_id, turn)

        if not result.evicted:
            return

        self.eviction_log.extend(result.events)
        trace.evicted.extend(result.evicted)
        trace.eviction_events.extend(result.events)

        # Mirror eviction marks into the store.
        if self.store is not None:
            for ev in result.events:
                self.store.mark_evicted(
                    self.session_id, ev.turn_id, ev.policy, ev.reason
                )

        # PROD PATTERN: Conversation Summarisation
        if self.enable_summary:
            self.summary = self.summariser.fold(result.evicted, self.summary)
            if self.store is not None and self.summary is not None:
                self.store.upsert_summary(self.session_id, self.summary)
            trace.summarised = True

        # PROD PATTERN: Semantic Memory via Vector Recall
        if self.enable_semantic:
            for t in result.evicted:
                # Add to the in-memory layer (computes the embedding).
                pre_size = self.semantic.size()
                self.semantic.add(t)
                if self.store is not None and self.semantic.size() > pre_size:
                    # The vector we just appended is the last one.
                    vec = self.semantic._vectors[-1]
                    self.store.add_vector(t.turn_id, self.session_id, vec)
