"""
RAG Chain — Phase 3

Assembles the full conversational RAG pipeline using LangChain Expression Language (LCEL).

Architecture (three chained components):

  1. history_aware_retriever
     Input:  user question + chat history
     Output: top-k relevant Document chunks
     How:    Rewrites follow-up questions into standalone queries before retrieval.
             Without this, "What else does it say?" would retrieve random chunks.

  2. qa_chain (create_stuff_documents_chain)
     Input:  user question + retrieved documents + chat history
     Output: grounded answer string
     How:    "Stuffs" all retrieved docs into the context window and generates answer.

  3. rag_chain (create_retrieval_chain)
     Wires 1 and 2 together into a single invokable pipeline.

Usage:
    retriever = get_retriever(qdrant_client)
    chain = build_rag_chain(retriever)

    response = chain.invoke({
        "input": "What is the main contribution?",
        "chat_history": [],  # or list of BaseMessage objects
    })

    answer = response["answer"]       # Generated text string
    context = response["context"]     # List of retrieved Document objects (for citations)
"""

import re

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from app.config import settings
from app.core.generation.prompts import get_contextualize_q_prompt, get_qa_prompt


def trim_history(messages: list) -> list:
    """
    Keep only the most recent `max_history_messages` turns so the prompt stays
    bounded no matter how long a conversation runs (prevents unbounded prompt
    growth → rising latency and cost over a long chat).
    """
    cap = settings.max_history_messages
    if cap and len(messages) > cap:
        return messages[-cap:]
    return messages


# Question intents that justify a longer answer (and thus a larger token budget).
# Checked first and mutually exclusive with brief cues (see dynamic_num_predict),
# so "explain what X is" is treated as detailed, not detailed-then-halved.
_DETAILED_CUES = (
    "explain", "describe", "compare", "contrast", "discuss", "summarize", "summarise",
    "elaborate", "walk through", "how does", "how do", "why", "analyze", "analyse",
    "list", "outline", "overview", "in detail", "step by step",
)
# Question shapes that usually want a short, factual answer.
_BRIEF_CUES = (
    "what is", "what are", "who", "when", "where", "define", "name the", "name all",
    "how many", "how much", "which", "is", "are", "does", "did", "can",
)


def _matches_any(text: str, cues: tuple[str, ...]) -> bool:
    """True if any cue appears in ``text`` on word boundaries.

    Word boundaries matter: a plain substring test makes "is" match inside "basis"
    and "can" inside "scan", misclassifying the question. ``\\b`` anchors each cue
    to whole words/phrases.
    """
    return any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in cues)


def dynamic_num_predict(question: str) -> int:
    """
    Choose a generation token budget that scales *continuously* with the question
    instead of snapping to a few fixed tiers. Three signals combine:

      - length:    longer questions tend to want longer answers (per-word base)
      - intent:    'explain/compare/list' widen the budget (x1.8); 'what is/define/
                   yes-no' narrow it (x0.5)
      - structure: each extra clause beyond the first ('?', ' and ', ';', list/all)
                   adds headroom for that part

    The result is clamped to [llm_num_predict_min, llm_num_predict_max]. num_predict
    is a cap, not a target — the model still stops at its natural EOS, so a generous
    budget never lengthens a genuinely short answer.
    """
    q = question.strip().lower()
    words = len(q.split())

    budget: float = settings.llm_num_predict_min + words * settings.llm_num_predict_per_word

    # Detailed wins outright — the two are mutually exclusive so a detailed
    # question is never also halved as "brief".
    if _matches_any(q, _DETAILED_CUES):
        budget *= 1.8
    elif _matches_any(q, _BRIEF_CUES):
        budget *= 0.5

    # Extra clauses beyond the first each get their own headroom.
    extra_parts = (
        max(0, q.count("?") - 1)
        + q.count(" and ")
        + q.count(";")
        + len(re.findall(r"\b(?:list|enumerate|all|each)\b", q))
    )
    budget += extra_parts * 128

    return int(max(settings.llm_num_predict_min, min(settings.llm_num_predict_max, budget)))


def context_aware_num_predict(question: str, context_docs: list) -> int:
    """
    Refine the question-based budget once retrieval has happened.

    dynamic_num_predict() only sees the question. After retrieval we also know how
    much relevant context came back: a question backed by a lot of retrieved
    material may warrant a longer synthesis, so we add headroom proportional to the
    retrieved character count (capped, then clamped to the global max).

    Computed *after* retrieval — the chain binds this onto the LLM for the
    generation step (see build_rag_chain).
    """
    budget = dynamic_num_predict(question)
    context_chars = sum(len(getattr(d, "page_content", "")) for d in context_docs)
    bonus = min(
        (context_chars // 1000) * settings.context_budget_per_1k_chars,
        settings.context_budget_max_bonus,
    )
    return int(min(settings.llm_num_predict_max, budget + bonus))


def get_llm(num_predict: int | None = None) -> ChatOllama:
    """
    Return the configured local LLM (served by Ollama).

    temperature=0 is critical for factual accuracy — never increase this for
    an academic chatbot. num_predict caps the generated answer length; pass a
    per-question value from dynamic_num_predict(). keep_alive holds the model in
    memory so back-to-back questions skip the cold reload.
    """
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=num_predict if num_predict is not None else settings.llm_num_predict_default,
        keep_alive=settings.llm_keep_alive,
        # ChatOllama has no top-level `timeout` field (it's silently dropped under
        # extra="ignore"); the request timeout must go to the underlying client.
        client_kwargs={"timeout": settings.llm_request_timeout},
    )


def build_rag_chain(retriever, num_predict: int | None = None):
    """
    Build the conversational RAG chain.

    Args:
        retriever: A LangChain VectorStoreRetriever (from get_retriever()).
        num_predict: Optional per-question token budget (see dynamic_num_predict()).

    Returns:
        A Runnable chain that accepts {"input": str, "chat_history": list}
        and returns {"answer": str, "context": list[Document]}.
    """
    llm = get_llm(num_predict)

    # Step 1: Rewrite follow-up questions into standalone queries
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=get_contextualize_q_prompt(),
    )

    # Step 2: Generate grounded answer from retrieved context.
    # The combine step runs AFTER retrieval, so its input dict already carries
    # "context" — we size the generation budget from it and bind it onto the LLM
    # for this call. Returning the Runnable (not its output) keeps token streaming
    # intact: LangChain invokes/streams the returned stuff-documents chain.
    def _qa_with_context_budget(inputs: dict):
        n = context_aware_num_predict(inputs["input"], inputs.get("context", []))
        return create_stuff_documents_chain(
            llm=llm.bind(num_predict=n),
            prompt=get_qa_prompt(),
        )

    qa_chain = RunnableLambda(_qa_with_context_budget)

    # Step 3: Combine retrieval + generation
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain,
    )

    # Step 4: Trim chat_history before it reaches the retriever/LLM so the prompt
    # stays bounded over long conversations.
    return RunnablePassthrough.assign(
        chat_history=lambda x: trim_history(x.get("chat_history", []))
    ) | rag_chain
