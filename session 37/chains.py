"""
Session 37 — chains.py

The heart of the session. Every chain here is built with LCEL — the
LangChain Expression Language — which uses the pipe operator `|` to wire
components together left to right:

    prompt | model | output_parser

Read that as: fill the prompt, send it to the model, parse the reply. Each
piece is a Runnable, and `|` connects the output of one to the input of
the next — exactly like a Unix pipe, or a factory assembly line.

Three chains live here:
  build_template_chain()  — the smallest possible LCEL chain (Demo 2)
  build_summary_chain()   — summarisation, streamed (Demo 4)
  build_qa_chain()        — retrieval-augmented document Q&A (Demo 3/5)
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from model_factory import get_chat_model
from prompts import QA_SYSTEM, SUMMARISER_SYSTEM, TEMPLATE_DEMO_SYSTEM
from retriever import format_docs, get_retriever


# ---------------------------------------------------------------------
# Demo 2 — the smallest LCEL chain: prompt | model | parser
# ---------------------------------------------------------------------


def build_template_chain() -> Runnable:
    """A three-link chain that explains any topic in one sentence.

    Shows the canonical LCEL shape and prompt-variable substitution. The
    {topic} placeholder is filled at call time via chain.invoke({...}).
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TEMPLATE_DEMO_SYSTEM),
            ("human", "Explain this topic in one sentence: {topic}"),
        ]
    )
    model = get_chat_model()
    parser = StrOutputParser()
    return prompt | model | parser


# ---------------------------------------------------------------------
# Demo 4 — summarisation chain (built to be streamed)
# ---------------------------------------------------------------------


def build_summary_chain() -> Runnable:
    """Summarise a long {document} into a short paragraph.

    Same three-link shape. The interesting part is at call time: we use
    chain.stream() instead of chain.invoke() so tokens arrive one at a
    time — the right UX for a long answer.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARISER_SYSTEM),
            ("human", "Summarise the following text:\n\n{document}"),
        ]
    )
    model = get_chat_model()
    return prompt | model | StrOutputParser()


# ---------------------------------------------------------------------
# Demo 3 / 5 — document Q&A chain (retrieval-augmented)
# ---------------------------------------------------------------------


def build_qa_chain(k: int = 2) -> Runnable:
    """Retrieve relevant chunks, stuff them into a prompt, then answer.

    This is the canonical LCEL RAG chain. The dictionary at the front runs
    two branches in parallel:
      - "context"  : take the question -> retriever -> format_docs
      - "question" : pass the question straight through unchanged
    Both feed the prompt, which feeds the model, which feeds the parser.

    The whole Phase 4 pipeline — retrieve, stuff, generate — is these few
    lines once LangChain owns the wiring.
    """

    retriever = get_retriever(k=k)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    model = get_chat_model()

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
