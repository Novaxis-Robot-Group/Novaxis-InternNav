import json
import sys
from pathlib import Path


REALWORLD_DIR = Path(__file__).resolve().parents[2] / "scripts" / "realworld"
if str(REALWORLD_DIR) not in sys.path:
    sys.path.insert(0, str(REALWORLD_DIR))

from agent_long_term_memory import (  # noqa: E402
    JsonlGraphMemorySink,
    build_memory_candidate,
    build_retrieval_query,
)
from upper_agent import build_prompt, normalize_upper_agent_config  # noqa: E402


def test_unverified_subgoal_change_is_not_long_term_memory():
    previous = {"active_subgoal": "Go straight.", "current_place": "corridor"}
    unchanged = {
        "task_status": "running",
        "current_subgoal": "Go straight.",
        "memory": {"current_place": "corridor"},
    }
    assert (
        build_memory_candidate(
            previous,
            unchanged,
            run_name="run-a",
            frame_idx=10,
            task_instruction="Find the sofa.",
        )
        is None
    )

    changed = {
        "task_status": "running",
        "current_subgoal": "Turn right at the glass doors.",
        "memory": {"current_place": "glass door intersection"},
    }
    assert (
        build_memory_candidate(
            previous,
            changed,
            run_name="run-a",
            frame_idx=20,
            task_instruction="Find the sofa.",
        )
        is None
    )


def test_completed_subgoal_becomes_verified_success_memory():
    previous = {"active_subgoal": "Go straight.", "current_place": "corridor"}
    completed = {
        "task_status": "running",
        "current_subgoal": "Turn right at the glass doors.",
        "memory": {
            "current_place": "glass door intersection",
            "completed_subgoal": "Go straight.",
            "landmarks_seen": ["glass doors"],
        },
    }
    candidate = build_memory_candidate(
        previous,
        completed,
        run_name="run-a",
        frame_idx=20,
        task_instruction="Find the sofa.",
    )
    assert candidate["memory_types"] == ["success"]
    assert candidate["action"] == "Go straight."
    assert candidate["result"] == "subgoal completed: Go straight."
    assert candidate["outcome"] == "subgoal_completed"
    assert candidate["landmarks"] == ["glass doors"]


def test_failure_and_grounded_finding_are_typed_separately():
    failed = build_memory_candidate(
        {"active_subgoal": "Turn right."},
        {
            "task_status": "failed",
            "current_subgoal": "",
            "memory": {
                "current_place": "blind corner",
                "failure_reason": "wall blocks the turn",
            },
        },
        run_name="run-f",
        frame_idx=33,
        task_instruction="Inspect the corridor.",
    )
    assert failed["memory_types"] == ["failure"]
    assert failed["action"] == "Turn right."
    assert failed["failure_reason"] == "wall blocks the turn"
    assert failed["text"].startswith("memory_types=failure")

    ungrounded_finding = build_memory_candidate(
        {},
        {
            "task_status": "running",
            "task_feedback": {
                "findings": [{"description": "a damaged box", "evidence": ""}],
            },
        },
        run_name="run-f",
        frame_idx=34,
        task_instruction="Inspect the corridor.",
    )
    assert ungrounded_finding is None

    grounded_finding = build_memory_candidate(
        {},
        {
            "task_status": "running",
            "task_feedback": {
                "findings": [
                    {
                        "type": "damage",
                        "description": "a damaged box",
                        "location": "north corridor",
                        "evidence": "torn cardboard is visible",
                    }
                ],
            },
        },
        run_name="run-f",
        frame_idx=35,
        task_instruction="Inspect the corridor.",
    )
    assert grounded_finding["memory_types"] == ["finding"]
    assert grounded_finding["findings"][0]["description"] == "a damaged box"


def test_graph_sink_records_typed_success_relations(tmp_path):
    candidate = build_memory_candidate(
        {"active_subgoal": "Go straight."},
        {
            "task_status": "completed",
            "current_subgoal": "",
            "memory": {
                "current_place": "sofa area",
                "completed_subgoal": "Approach the gray sofa.",
                "landmarks_seen": ["gray sofa"],
            },
        },
        run_name="run-b",
        frame_idx=99,
        task_instruction="Find the gray sofa.",
    )
    path = tmp_path / "graph_events.jsonl"
    JsonlGraphMemorySink(path).append(candidate)
    event = json.loads(path.read_text(encoding="utf-8"))
    relations = {edge["relation"] for edge in event["edges"]}
    assert event["schema_version"] == 2
    assert event["memory_types"] == ["success"]
    assert {"CONTAINS", "SUCCEEDED_AT", "HAS_LANDMARK"} <= relations
    assert "RESULTED_IN" not in relations


def test_graph_sink_records_failure_and_finding_relations(tmp_path):
    candidate = build_memory_candidate(
        {"active_subgoal": "Turn right."},
        {
            "task_status": "failed",
            "memory": {
                "current_place": "blind corner",
                "failure_reason": "wall blocks the turn",
            },
            "task_feedback": {
                "findings": [
                    {
                        "description": "a damaged box",
                        "location": "blind corner",
                        "evidence": "torn cardboard is visible",
                    }
                ],
            },
        },
        run_name="run-c",
        frame_idx=40,
        task_instruction="Inspect the corridor.",
    )
    path = tmp_path / "graph_events.jsonl"
    JsonlGraphMemorySink(path).append(candidate)
    event = json.loads(path.read_text(encoding="utf-8"))
    relations = {edge["relation"] for edge in event["edges"]}
    assert event["memory_types"] == ["failure", "finding"]
    assert {"FAILED_AT", "FAILED_BECAUSE", "OBSERVED"} <= relations


def test_prompt_uses_compact_cross_experiment_memory():
    prompt = build_prompt(
        {"frame_idx": 5, "response": {}},
        [],
        {"instruction": "Go straight."},
        {"current_place": "corridor", "active_subgoal": "Go straight."},
        normalize_upper_agent_config({}),
        long_term_memories=[{"memory": "turn right at glass doors succeeded", "score": 0.8}],
    )
    assert "cross_experiment_memory" in prompt
    assert "turn right at glass doors succeeded" in prompt


def test_retrieval_query_stays_compact():
    query = build_retrieval_query(
        "Find the gray sofa.",
        {
            "current_place": "office corridor",
            "active_subgoal": "Go straight.",
            "visited_landmarks": ["glass doors", "exit sign"],
        },
        "Turn right.",
    )
    assert "office corridor" in query
    assert "glass doors" in query
    assert len(query) < 500
