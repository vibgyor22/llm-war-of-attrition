"""Tests for the LLM cache module."""
import tempfile
from pathlib import Path
import pytest
from src.cache.llm_cache import LLMCache, CacheMissError


def _make_cache(mode="read-write"):
    tmp = tempfile.mkdtemp()
    return LLMCache(Path(tmp) / "test.sqlite", mode=mode)


class TestLLMCache:
    def test_put_and_get(self):
        cache = _make_cache()
        key = cache.make_key("model-x", "sys", [{"role": "user", "content": "hi"}], 0.8, None, "HAWK")
        assert cache.get(key) is None
        resp = {"content": [{"type": "text", "text": "hello"}], "model": "model-x"}
        cache.put(key, "model-x", "HAWK", resp, 100, 50)
        got = cache.get(key)
        assert got is not None
        assert got["model"] == "model-x"

    def test_cache_key_deterministic(self):
        cache = _make_cache()
        k1 = cache.make_key("m", "sys", [{"role": "user", "content": "q"}], 0.9, None, "HAWK")
        k2 = cache.make_key("m", "sys", [{"role": "user", "content": "q"}], 0.9, None, "HAWK")
        assert k1 == k2

    def test_cache_key_role_collision(self):
        cache = _make_cache()
        k_hawk = cache.make_key("m", "sys", [{"role": "user", "content": "q"}], 0.9, None, "HAWK")
        k_dove = cache.make_key("m", "sys", [{"role": "user", "content": "q"}], 0.9, None, "DOVE")
        assert k_hawk != k_dove

    def test_read_only_miss_raises(self):
        cache = _make_cache(mode="read-only")
        key = cache.make_key("m", "s", [], 0.5, None, "TEST")
        with pytest.raises(CacheMissError):
            cache.get(key)

    def test_read_only_put_raises(self):
        cache = _make_cache(mode="read-only")
        key = "deadbeef"
        with pytest.raises(RuntimeError):
            cache.put(key, "m", "TEST", {}, 0, 0)

    def test_stats_empty(self):
        cache = _make_cache()
        s = cache.stats()
        assert s["total_cached"] == 0

    def test_stats_after_put(self):
        cache = _make_cache()
        key = cache.make_key("m", "s", [], 0.5, None, "X")
        cache.put(key, "m", "X", {}, 100, 50)
        s = cache.stats()
        assert s["total_cached"] == 1

    def test_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_CACHE_MODE", "read-write")
        cache = LLMCache.from_env(tmp_path / "c.sqlite")
        assert cache._mode == "read-write"

    def test_thread_safety(self):
        import concurrent.futures
        cache = _make_cache()
        def _worker(i):
            key = cache.make_key("m", "s", [{"role": "user", "content": str(i)}], 0.5, None, "HAWK")
            cache.put(key, "m", "HAWK", {"i": i}, 10, 5)
            return cache.get(key)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_worker, range(50)))
        assert all(r is not None for r in results)
