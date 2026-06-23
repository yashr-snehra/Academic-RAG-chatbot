"""doc_store: sync write + async read, exercised with tiny in-memory fakes (no Redis)."""

import asyncio

from app.core.memory import doc_store


class _FakeSyncRedis:
    def __init__(self):
        self.store: dict[str, dict] = {}

    def hset(self, key, mapping):
        self.store[key] = dict(mapping)


class _FakeAsyncRedis:
    def __init__(self, store: dict[str, dict]):
        self._store = store

    async def keys(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in self._store if k.startswith(prefix)]

    async def hgetall(self, key):
        # Redis returns all values as strings (decode_responses=True).
        return {k: str(v) for k, v in self._store[key].items()}


def test_record_writes_expected_fields():
    fake = _FakeSyncRedis()
    doc_store.record_document("attention", total_chunks=42, pages=11, client=fake)

    rec = fake.store["doc_meta:attention"]
    assert rec["name"] == "attention"
    assert rec["total_chunks"] == 42
    assert rec["pages"] == 11
    assert rec["ingested_at"].endswith("+00:00")  # ISO-8601 UTC


def test_get_documents_reads_and_sorts():
    fake = _FakeSyncRedis()
    doc_store.record_document("zeta", total_chunks=3, pages=2, client=fake)
    doc_store.record_document("alpha", total_chunks=5, pages=9, client=fake)

    docs = asyncio.run(doc_store.get_documents(_FakeAsyncRedis(fake.store)))
    assert [d["name"] for d in docs] == ["alpha", "zeta"]  # sorted
    assert docs[0]["total_chunks"] == 5
    assert docs[0]["pages"] == 9


def test_get_documents_empty_when_nothing_recorded():
    assert asyncio.run(doc_store.get_documents(_FakeAsyncRedis({}))) == []
