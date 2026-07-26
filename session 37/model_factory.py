"""
Session 37 — model_factory.py

One place that hands every chain a chat model. This is the only file that
knows whether we are talking to the real Anthropic API or to an offline
fake. Every chain calls get_chat_model() and never imports a provider
directly — so switching from fake to real is a single env-var flip.

    FAKE_LLM=1   -> RoutedFakeChatModel (offline, deterministic, no key)
    (otherwise)  -> ChatAnthropic       (real Claude call, needs a key)

PHASE 5 RULE — opener-phrase routing.
The offline fake decides what to "answer" by reading the FIRST line of the
system prompt, which always starts "You are a <role>." It never routes on
a topic word, because topic words leak across prompts. See prompts.py.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult



# Hard-disable LangChain's automatic LangSmith / hosted tracing.
# This session teaches observability via a hand-built callback handler;
# we never want background calls to an external tracing service,
# regardless of any LANGCHAIN_TRACING_V2 / LANGSMITH_API_KEY the user has
# set in their shell or .env. Force it off before any chain runs.
for _flag in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
    os.environ[_flag] = "false"
for _key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
             "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"):
    os.environ.pop(_key, None)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# A fixed, hand-written summary the fake returns when it is asked to
# summarise. Long enough to make the streaming demo look real.
_CANNED_SUMMARY = (
    "NovaDesk is a browser-based customer-support platform for small and "
    "mid-size software teams, built around a shared inbox, a help-centre, "
    "and an AI reply assistant. It sells three plans — a free Starter "
    "plan, a per-agent Team plan, and a Pro plan that adds the AI "
    "assistant and a 99.9 percent uptime guarantee. Every paid plan has a "
    "30-day full-refund window. NovaDesk connects to Slack, Microsoft "
    "Teams, and email, and exposes a REST API on the Pro plan. This "
    "year's roadmap focuses on multilingual replies, deeper per-agent "
    "reporting, and a mobile app for agents."
)


class RoutedFakeChatModel(BaseChatModel):
    """A deterministic offline chat model for the demos.

    It inspects the system prompt's opener phrase and the human message,
    then returns a sensible canned answer. It counts how many times it is
    actually invoked, which lets the caching demo prove a cache hit
    (a cached call never reaches _generate, so the counter does not move).
    """

    # Pydantic field so BaseChatModel (a pydantic model) accepts it.
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "routed-fake-chat-model"

    # -- the routing brain -------------------------------------------------

    def _route(self, messages: List[BaseMessage]) -> str:
        system_text = ""
        human_text = ""
        for m in messages:
            if m.type == "system":
                system_text += str(m.content).lower() + "\n"
            elif m.type == "human":
                human_text += str(m.content).lower() + "\n"

        # Opener-phrase routes — never topic words.
        if "you are a summariser" in system_text:
            return _CANNED_SUMMARY

        if "you are a document q&a assistant" in system_text:
            return self._fake_answer_from_context(human_text)

        if "you are a prompt-template demo assistant" in system_text:
            # Echo back whatever topic the template filled in, to show
            # that variable substitution actually happened. The topic is
            # the text after the final colon in the filled human message.
            topic = human_text.rsplit(":", 1)[-1].strip() if ":" in human_text else human_text.strip()
            return f"A quick, plain-English take on {topic}: it is a tool that makes building LLM apps faster."

        return "(routed-fake) no canned route matched this system prompt."

    @staticmethod
    def _fake_answer_from_context(human_text: str) -> str:
        """Pick an answer using keywords in the QUESTION first.

        The Q&A prompt is shaped "Context:\n...\n\nQuestion: <q>". A real
        model answers the question, so the fake routes on the question text
        (after the final 'question:'), falling back to the whole blob. This
        avoids a stray keyword in the stuffed context hijacking the route.
        Deterministic so the Q&A demo is identical every run.
        """

        if "question:" in human_text:
            t = human_text.rsplit("question:", 1)[-1]
        else:
            t = human_text
        if "refund" in t:
            return (
                "NovaDesk offers a 30-day refund window on every paid plan. "
                "If you cancel within 30 days of your first payment, you are "
                "refunded in full to your original payment method within five "
                "business days. [source: pricing_and_refunds.md]"
            )
        if "slack" in t or "integrat" in t or "teams" in t:
            return (
                "NovaDesk integrates with Slack, Microsoft Teams, and email, "
                "and offers a REST API on the Pro plan. "
                "[source: integrations.md]"
            )
        if "uptime" in t or "99.9" in t or "service credit" in t:
            return (
                "The Pro plan SLA guarantees 99.9 percent monthly uptime. If "
                "uptime falls below that in a calendar month, affected Pro "
                "customers get service credits on the next invoice. "
                "[source: sla_and_uptime.md]"
            )
        if "roadmap" in t or "mobile" in t or "language" in t or "multilingual" in t:
            return (
                "This year's roadmap covers multilingual AI replies (Spanish, "
                "German, Hindi), deeper per-agent reporting, and a mobile app "
                "planned for the final quarter. [source: roadmap.md]"
            )
        return (
            "I could not find that in the provided NovaDesk documents, so I "
            "won't guess. [no matching source]"
        )

    # -- BaseChatModel plumbing -------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        text = self._route(messages)
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.call_count += 1
        text = self._route(messages)
        # Stream word by word so the streaming demo shows real incremental
        # output rather than one big blob.
        words = text.split(" ")
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            if run_manager:
                run_manager.on_llm_new_token(piece)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def is_fake() -> bool:
    """True when we are running in offline FAKE_LLM mode."""

    return os.environ.get("FAKE_LLM", "") == "1"


def get_chat_model(model: str = DEFAULT_MODEL) -> BaseChatModel:
    """Return the chat model every chain in this session uses.

    The whole point of this function: chains depend on the BaseChatModel
    interface, not on a provider. Fake and real are interchangeable.
    """

    if is_fake():
        return RoutedFakeChatModel()

    # Real mode — import lazily so the offline path never needs the
    # provider package installed.
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "langchain-anthropic is not installed. Run "
            "'pip install langchain-anthropic' or set FAKE_LLM=1 to run "
            "offline."
        ) from exc

    return ChatAnthropic(model=model, temperature=0, max_tokens=1024)
