"""Post-retrieval token budget + proof that the chain restructure keeps streaming."""

import asyncio
import itertools

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.core.generation import chain as chain_mod
from app.core.generation.chain import build_rag_chain, context_aware_num_predict, dynamic_num_predict


def _doc(n_chars: int) -> Document:
    return Document(page_content="x" * n_chars, metadata={"source_file": "p", "page_number": 1})


def test_no_context_equals_question_only_budget():
    q = "Explain the methodology in detail"
    assert context_aware_num_predict(q, []) == dynamic_num_predict(q)


def test_context_adds_proportional_bonus():
    # "hi": 1 word -> 128 + 24 = 152; 2000 chars -> (2000//1000)*80 = 160 bonus.
    assert context_aware_num_predict("hi", [_doc(2000)]) == 152 + 160


def test_bonus_is_capped():
    huge = context_aware_num_predict("hi", [_doc(10_000_000)])
    assert huge == min(settings.llm_num_predict_max, 152 + settings.context_budget_max_bonus)


def test_never_exceeds_global_max():
    budget = context_aware_num_predict("explain and compare and discuss " * 50, [_doc(5_000_000)])
    assert budget <= settings.llm_num_predict_max


def test_chain_still_streams_after_restructure(monkeypatch):
    """RunnableLambda-returns-Runnable must preserve token streaming, or /chat/stream breaks."""
    fake = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="hello world from context")]))
    monkeypatch.setattr(chain_mod, "get_llm", lambda num_predict=None: fake)

    stub_retriever = RunnableLambda(lambda q: [_doc(500)])
    rag_chain = build_rag_chain(stub_retriever)

    async def _run():
        tokens = []
        async for ck in rag_chain.astream({"input": "what is X?", "chat_history": []}):
            if ck.get("answer"):
                tokens.append(ck["answer"])
        return tokens

    tokens = asyncio.run(_run())
    assert "".join(tokens) == "hello world from context"
    assert len(tokens) >= 2  # genuinely streamed in pieces, not one blob
