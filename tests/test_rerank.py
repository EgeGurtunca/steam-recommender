from recs import llm
from recs.data import Catalog
from recs.rerank import LLMRerank, schema


class FakeHybrid:
    def __init__(self, cands):
        self.cands = cands

    def candidates(self, library):
        return [a for a in self.cands if a not in library]


CAT = Catalog({a: f"Game {a}" for a in range(1, 30)}, {a: ["Action"] for a in range(1, 30)},
              {a: "2020-01-01" for a in range(1, 30)})


def test_schema_only_allows_candidates():
    s = schema([5, 6, 7], 2)
    picks = s["properties"]["picks"]
    assert picks["minItems"] == picks["maxItems"] == 2
    assert picks["items"]["properties"]["app_id"]["enum"] == [5, 6, 7]


def test_duplicates_and_strangers_are_replaced_from_hybrid_order():
    fake = lambda prompt, sch: {"picks": [{"app_id": 5, "reason": "a"}, {"app_id": 5, "reason": "b"},
                                          {"app_id": 999, "reason": "c"}]}
    r = LLMRerank(FakeHybrid([3, 4, 5, 6]), CAT, generate=fake)
    out = r.recommend_with_reasons({1: 10.0}, k=3)
    assert [a for a, _ in out] == [5, 3, 4] and out[0][1] == "a"


def test_llm_failure_falls_back_and_counts():
    def boom(prompt, sch):
        raise TimeoutError
    r = LLMRerank(FakeHybrid([3, 4, 5]), CAT, generate=boom)
    assert r.recommend({1: 10.0}, k=2) == [3, 4] and r.failures == 1


def test_ollama_is_stopped_with_children_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "IS_WINDOWS", True)
    monkeypatch.setattr(llm.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    llm.stop(type("Proc", (), {"pid": 42})())
    assert calls == [["taskkill", "/PID", "42", "/T", "/F"]]
