"""
Session 38 — prebuilt_agent.py

THE PREBUILT PATH. Same lesson as S37's "pure Python vs LangChain": you just
hand-wired a graph (graph.py). Now meet the parts LangGraph ships so you don't
have to wire a tool-calling agent by hand.

Three prebuilt pieces, all from langgraph.prebuilt:

  MessagesState   — a ready-made State whose single box, `messages`, uses the
                    `add_messages` reducer. Every node that returns
                    {"messages": [...]} APPENDS to the conversation instead of
                    erasing it. (This is the reducer idea from state.py, but
                    the most famous reducer of all.)

  ToolNode        — a ready-made NODE that looks at the last message, runs any
                    tool calls the model asked for, and appends the results as
                    messages. You don't write the tool-dispatch loop yourself.

  tools_condition — a ready-made CONDITIONAL EDGE. After the agent speaks, it
                    routes to "tools" if the model asked to call a tool, or to
                    END if the model gave a final answer.

Wire those three together and you have a real ReAct agent — the same
decide -> act -> observe loop you built by hand in S29 — in about a dozen
lines. The cycle here is agent -> tools -> agent, and it ends when the model
stops asking for tools.

ONE-LINER NOTE: `langchain.agents.create_agent(model, tools=[...])` builds this
exact graph for you in a single call. (In older LangGraph it was
`langgraph.prebuilt.create_react_agent`, now deprecated and moved.) We build it
from the pieces here so you can SEE that the one-liner is just nodes + a
conditional edge + the add_messages reducer — nothing you don't already know.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import search_tools


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information and return the top results
    as text. Use this whenever the user asks about facts you are unsure of."""
    hits = search_tools.web_search(query, max_results=4)
    if not hits:
        return "No results found."
    return "\n\n".join(
        f"{h.get('title', '')}\n{h.get('content', '')}\nURL: {h.get('url', '')}"
        for h in hits
    )


TOOLS = [web_search]


def build_prebuilt_agent():
    """Assemble a ReAct tool-calling agent from the prebuilt pieces.

    Returns a compiled graph. Needs a real ANTHROPIC_API_KEY to .invoke()
    (the model is only called at run time), but compiles and draws offline.
    """
    import llm

    model = llm.get_chat_model().bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict:
        # The model reads the running conversation and replies. Returning the
        # reply under "messages" APPENDS it via the add_messages reducer.
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))         # prebuilt tool-runner node
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # prebuilt fork: tools? or END
    graph.add_edge("tools", "agent")                 # the ReAct loop
    return graph.compile()


def prebuilt_graph_ascii() -> str:
    """Draw the prebuilt agent's graph WITHOUT calling the model.

    Constructing ChatAnthropic does not need a key (the key is only checked at
    call time), so this works offline for the walkthrough.
    """
    import os

    # Build the model object directly so we don't trip llm.get_chat_model()'s
    # no-key guard — drawing the graph never calls the model.
    from langchain_anthropic import ChatAnthropic

    model_name = os.environ.get("S38_MODEL", "claude-haiku-4-5-20251001")
    model = ChatAnthropic(model=model_name).bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    app = graph.compile()
    try:
        return app.get_graph().draw_ascii()
    except ImportError:
        return ("Install `grandalf` to draw the graph.\n"
                "Shape: START -> agent -> (tools_condition) -> {tools -> agent | END}")
