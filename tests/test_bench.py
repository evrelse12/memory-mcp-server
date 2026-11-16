# tests/test_bench.py
"""Offline unit tests for the benchmark harness (no services, no dataset)."""
import pytest

from bench import datasets, harness


def test_parse_ts():
    assert datasets.parse_ts("2023/05/20 (Sat) 12:00") is not None
    assert datasets.parse_ts("1:32 pm on 8 May, 2023") is not None
    assert datasets.parse_ts("nonsense") is None
    assert datasets.parse_ts(1700000000) == 1700000000


def test_detect_format():
    assert datasets.detect_format([{"haystack_sessions": []}]) == "longmemeval"
    assert datasets.detect_format([{"conversation": {}, "qa": []}]) == "locomo"
    with pytest.raises(ValueError):
        datasets.detect_format([{"foo": 1}])


def test_load_longmemeval():
    raw = [{"question_id": "q1", "question_type": "knowledge-update",
            "question": "Where does the user work?", "answer": "Google",
            "question_date": "2023/05/20 (Sat) 12:00",
            "haystack_dates": ["2023/05/01 (Mon) 10:00"],
            "haystack_session_ids": ["sess-1"],
            "haystack_sessions": [[{"role": "user", "content": "I joined Google"}]],
            "answer_session_ids": ["sess-1"]},
           {"question_id": "q2_abs", "question_type": "multi-session",
            "question": "Cat name?", "answer": "unknown",
            "haystack_sessions": [[]], "answer_session_ids": []}]
    items = datasets.load_longmemeval(raw)
    q = items[0].questions[0]
    assert q.kind == "knowledge-updates" and q.evidence == ["sess-1"]
    assert items[0].sessions[0].sid == "sess-1" and items[0].sessions[0].ts > 0
    assert items[1].questions[0].abstention is True
    assert items[1].questions[0].kind == "abstention"


def test_load_locomo():
    raw = [{"sample_id": "conv-1",
            "conversation": {"session_1": [{"speaker": "Alice", "text": "hi"}],
                             "session_1_date_time": "1:32 pm on 8 May, 2023"},
            "qa": [{"question": "Who said hi?", "answer": "Alice", "category": 4},
                   {"question": "Unanswerable?", "adversarial_answer": "no info",
                    "category": 5}]}]
    items = datasets.load_locomo(raw)
    assert items[0].sessions[0].sid == "session_1"
    assert items[0].sessions[0].turns[0].role == "user"
    qs = items[0].questions
    assert qs[0].kind == "single-hop"
    assert qs[1].abstention is True and qs[1].answer == "no info"


def test_yesno():
    assert harness._yesno("Yes, correct") is True
    assert harness._yesno("no") is False
    assert harness._yesno("maybe") is None


def test_aggregate():
    r = harness.Result
    rs = [r("i", "a", "k", False, "memory", True, True, 3, 10.0, 5.0),
          r("i", "b", "k", False, "memory", False, False, 3, 20.0, 5.0),
          r("i", "c", "k", False, "memory", None, None, 0, 30.0, 5.0)]
    rep = harness.aggregate(rs)["memory"]
    assert rep["n"] == 3 and rep["judged"] == 2 and rep["judge_errors"] == 1
    assert rep["accuracy"] == 0.5
    assert rep["recall_at_k"] == 0.5
    assert rep["by_kind"]["k"]["accuracy"] == 0.5
