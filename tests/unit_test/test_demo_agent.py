import sys
from pathlib import Path


REALWORLD_DIR = Path(__file__).resolve().parents[2] / "scripts" / "realworld"
if str(REALWORLD_DIR) not in sys.path:
    sys.path.insert(0, str(REALWORLD_DIR))

from demo_agent import (  # noqa: E402
    activate_demo_library,
    constrain_demo_agent_output,
    control_demo_agent,
    delete_demo_library,
    ensure_demo_step_reference,
    get_demo_state,
    load_demo_libraries,
    parse_navigation_steps,
    upsert_demo_library,
)
from experiment_visualizer import create_viewer_app  # noqa: E402
from runtime_config import load_runtime_config, save_runtime_config  # noqa: E402
import upper_agent  # noqa: E402
from upper_agent import (  # noqa: E402
    build_demo_execution_evidence,
    select_motion_context_frames,
    summarize_low_level_motion,
)
from PIL import Image  # noqa: E402


def test_parse_navigation_steps_supports_delimiters_and_atomic_spaces():
    assert parse_navigation_steps("前进 左转 停止") == ["Go straight.", "Turn left.", "Stop."]
    assert parse_navigation_steps("Go straight; Turn right, Stop near the sofa") == [
        "Go straight",
        "Turn right",
        "Stop near the sofa",
    ]
    assert parse_navigation_steps('"Go straight to the doors" "Turn left"') == [
        "Go straight to the doors",
        "Turn left",
    ]
    assert parse_navigation_steps("Go straight to the doors") == ["Go straight to the doors"]
    assert parse_navigation_steps("go straight turn right stop") == [
        "Go straight.",
        "Turn right.",
        "Stop.",
    ]


def test_motion_context_samples_step_start_intervals_and_latest(tmp_path):
    for frame_idx in range(10):
        Image.new("RGB", (16, 12), (frame_idx, 0, 0)).save(
            tmp_path / f"frame_{frame_idx:06d}_rgb.jpg"
        )
    sampled = select_motion_context_frames(
        tmp_path,
        "frame_000000_rgb.jpg",
        "frame_000009_rgb.jpg",
        frame_count=4,
    )
    assert [item["frame_idx"] for item in sampled] == [0, 3, 6, 9]


def test_trajectory_evidence_and_turn_gate_prevent_early_advance(tmp_path):
    assert summarize_low_level_motion(
        {"trajectory": [[0.0, 0.0], [0.5, -0.2], [0.8, -0.8]]}
    )["planned_motion"] == "right"

    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": True, "upper_agent": {}})
    library = {
        "id": "left-route",
        "name": "Left route",
        "scene": "Lab",
        "notes": "",
        "commands": ["Go straight, then turn left.", "Go straight."],
    }
    activate_demo_library(config_path, library)
    for frame_idx in (1, 5):
        image_name = f"frame_{frame_idx:06d}_rgb.jpg"
        Image.new("RGB", (16, 12), (frame_idx, 0, 0)).save(tmp_path / image_name)
        (tmp_path / f"frame_{frame_idx:06d}_waypoint.json").write_text(
            __import__("json").dumps(
                {
                    "frame_idx": frame_idx,
                    "instruction": library["commands"][0],
                    "response": {"trajectory": [[0.0, 0.0], [0.8, 0.01], [1.5, 0.02]]},
                }
            )
        )
    frames = [
        {"frame_idx": 1, "image_file": "frame_000001_rgb.jpg"},
        {"frame_idx": 5, "image_file": "frame_000005_rgb.jpg"},
    ]
    evidence = build_demo_execution_evidence(tmp_path, frames, library["commands"][0])
    assert evidence["required_turn"] == "left"
    assert evidence["has_required_turn_output"] is False

    constrained, changed = constrain_demo_agent_output(
        config_path,
        {"task_status": "running", "demo_step_decision": "advance", "visual_evidence": "Moved."},
        execution_evidence=evidence,
    )
    assert changed is False
    assert constrained["demo_agent"]["step_index"] == 0
    assert constrained["demo_agent"]["turn_transition_gate_blocked"] is True

    planned_but_not_completed = {
        **evidence,
        "has_required_turn_output": True,
        "matching_turn_output_count": 2,
    }
    constrained, _ = constrain_demo_agent_output(
        config_path,
        {
            "task_status": "running",
            "demo_step_decision": "advance",
            "execution_assessment": {
                "subgoal_completed": False,
                "turn_started": True,
                "turn_completed": False,
                "observed_turn_direction": "left",
                "completion_confidence": 0.55,
            },
        },
        execution_evidence=planned_but_not_completed,
    )
    assert constrained["demo_agent"]["step_index"] == 0
    assert constrained["demo_agent"]["turn_transition_gate_blocked"] is True


def test_balanced_demo_mode_advances_after_supported_turn_clip(tmp_path):
    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": True, "upper_agent": {}})
    activate_demo_library(
        config_path,
        {
            "id": "balanced-route",
            "name": "Balanced route",
            "scene": "Office",
            "notes": "",
            "commands": ["Go straight and turn right.", "Continue straight."],
        },
    )
    evidence = {
        "required_turn": "right",
        "has_required_turn_output": True,
        "matching_turn_output_count": 13,
        "clip_ended_with_stop": True,
        "endpoint_visual_change_score": 0.32,
        "completion_confidence_threshold": 0.75,
        "transition_mode": "balanced",
    }
    output = {
        "task_status": "running",
        "demo_step_decision": "hold",
        "execution_assessment": {
            "subgoal_completed": False,
            "turn_started": True,
            "turn_completed": False,
            "observed_turn_direction": "right",
            "completion_confidence": 0.45,
        },
        "visual_evidence": "The endpoint view is a new office area.",
    }

    constrained, changed = constrain_demo_agent_output(
        config_path,
        output,
        run_name="run-a",
        frame_idx=75,
        image_file="frame_000075_rgb.jpg",
        execution_evidence=evidence,
    )

    assert changed is True
    assert constrained["demo_agent"]["step_index"] == 1
    assert constrained["demo_agent"]["decision"] == "advance"
    assert constrained["demo_agent"]["balanced_transition_supported"] is True
    assert constrained["current_subgoal"] == "Continue straight."


def test_balanced_demo_mode_accepts_strong_temporal_evidence_when_vlm_is_uncertain(tmp_path):
    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": True, "upper_agent": {}})
    activate_demo_library(
        config_path,
        {
            "id": "temporal-route",
            "name": "Temporal route",
            "scene": "Office",
            "notes": "",
            "commands": ["Go straight and turn right.", "Turn left."],
        },
    )
    constrained, _ = constrain_demo_agent_output(
        config_path,
        {
            "task_status": "running",
            "demo_step_decision": "hold",
            "execution_assessment": {
                "subgoal_completed": False,
                "turn_started": False,
                "turn_completed": False,
                "observed_turn_direction": "none",
                "completion_confidence": 0.15,
            },
        },
        execution_evidence={
            "required_turn": "right",
            "has_required_turn_output": True,
            "matching_turn_output_count": 16,
            "matching_turn_segment_count": 3,
            "clip_ended_with_stop": True,
            "endpoint_visual_change_score": 0.2053,
            "completion_confidence_threshold": 0.75,
            "transition_mode": "balanced",
        },
    )

    assert constrained["demo_agent"]["step_index"] == 1
    assert constrained["demo_agent"]["strong_temporal_transition"] is True
    assert constrained["current_subgoal"] == "Turn left."


def test_demo_library_lifecycle_and_order_constraint(tmp_path):
    store_path = tmp_path / "demo_agent_libraries.json"
    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": False, "upper_agent": {"api_key": "secret"}})
    library = upsert_demo_library(
        store_path,
        {
            "name": "Office route",
            "scene": "Office A",
            "notes": "Start at lift",
            "commands_text": "Go straight to the doors; Turn right; Stop near the sofa",
        },
    )
    library = upsert_demo_library(store_path, {**library, "notes": "Start at the east lift"})
    assert len(load_demo_libraries(store_path)["libraries"]) == 1
    assert library["notes"] == "Start at the east lift"

    activated = activate_demo_library(config_path, library)
    state = get_demo_state(activated)
    assert state["status"] == "running"
    assert activated["instruction"] == "Go straight to the doors"
    assert activated["service_enabled"] is True
    assert activated["upper_agent"]["hard_reset_requested"] is True

    reference = ensure_demo_step_reference(config_path, "run-a", 4, "frame_000004_rgb.jpg")
    assert reference["frame_idx"] == 4
    assert reference["image_file"] == "frame_000004_rgb.jpg"

    invented = {
        "task_status": "running",
        "current_subgoal": "Fly to the roof",
        "demo_step_index": 2,
        "demo_step_decision": "advance",
    }
    constrained, changed = constrain_demo_agent_output(
        config_path,
        invented,
        run_name="run-a",
        frame_idx=12,
        image_file="frame_000012_rgb.jpg",
    )
    assert changed is True
    assert constrained["current_subgoal"] == "Turn right"
    assert constrained["demo_agent"]["step_index"] == 1
    next_state = get_demo_state(load_runtime_config(config_path))
    assert next_state["step_started_step_index"] == 1
    assert next_state["step_started_frame_idx"] == 12
    assert next_state["step_started_image_file"] == "frame_000012_rgb.jpg"

    control_demo_agent(config_path, "pause")
    assert load_runtime_config(config_path)["service_enabled"] is False
    resumed = control_demo_agent(config_path, "resume")
    assert resumed["instruction"] == "Turn right"
    assert resumed["upper_agent"]["hard_reset_requested"] is True

    control_demo_agent(config_path, "stop")
    delete_demo_library(store_path, library["id"], runtime_config_path=config_path)
    assert load_demo_libraries(store_path)["libraries"] == []


def test_demo_agent_http_api(tmp_path):
    log_dir = tmp_path / "runs"
    log_dir.mkdir()
    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": False, "upper_agent": {}})
    app = create_viewer_app(log_dir, runtime_config_path=config_path)
    client = app.test_client()

    created = client.post(
        "/api/demo-agent/libraries",
        json={"name": "Lobby", "scene": "Floor 3", "notes": "lift start", "commands_text": "前进 左转 停止"},
    )
    assert created.status_code == 200
    library = created.get_json()["library"]
    assert library["commands"] == ["Go straight.", "Turn left.", "Stop."]
    assert client.post(f"/api/demo-agent/activate/{library['id']}").status_code == 200
    active = client.get("/api/demo-agent/libraries").get_json()["active"]
    assert active["library_name"] == "Lobby"
    assert client.delete(f"/api/demo-agent/libraries/{library['id']}").status_code == 400
    assert client.post("/api/demo-agent/control", json={"action": "stop"}).status_code == 200
    assert client.delete(f"/api/demo-agent/libraries/{library['id']}").status_code == 200

    page = client.get("/")
    assert page.status_code == 200
    assert b"Demo Agent" in page.data
    assert b"demoLibraryForm" in page.data


def test_upper_agent_can_only_publish_ordered_library_commands(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "runtime.json"
    save_runtime_config(
        config_path,
        {
            "service_enabled": True,
            "upper_agent": {
                "enabled": True,
                "auto_apply_instruction": True,
                "pause_policy_while_thinking": False,
                "api_key": "test-only",
                "min_seconds_between_calls": 0,
            },
        },
    )
    library = {
        "id": "route-1",
        "name": "Fixed route",
        "scene": "Lab",
        "notes": "",
        "commands": ["Go straight to door A.", "Turn right at door A.", "Stop near desk B."],
    }
    activate_demo_library(config_path, library)

    model_outputs = iter(
        [
            {
                "task_status": "running",
                "navigation_phase": "advance",
                "current_subgoal": "Invented unsafe shortcut.",
                "demo_step_index": 2,
                "demo_step_decision": "advance",
                "execution_assessment": {
                    "subgoal_completed": True,
                    "forward_completed": True,
                    "turn_started": False,
                    "turn_completed": False,
                    "observed_turn_direction": "none",
                    "completion_confidence": 0.95,
                    "evidence_frame_indices": [0],
                },
                "visual_evidence": "Door A is ahead.",
                "memory": {},
            },
            {
                "task_status": "running",
                "navigation_phase": "turn",
                "current_subgoal": "Another invented command.",
                "demo_step_index": 2,
                "demo_step_decision": "advance",
                "execution_assessment": {
                    "subgoal_completed": True,
                    "forward_completed": True,
                    "turn_started": True,
                    "turn_completed": True,
                    "observed_turn_direction": "right",
                    "completion_confidence": 0.95,
                    "evidence_frame_indices": [0, 1],
                },
                "visual_evidence": "Door A was reached.",
                "memory": {},
            },
        ]
    )

    def fake_call(*_args, **_kwargs):
        return __import__("json").dumps(next(model_outputs)), {"usage": {}}, 0.01

    monkeypatch.setattr(upper_agent, "call_qwen_vl", fake_call)
    for frame_idx in range(2):
        image_name = f"frame_{frame_idx:06d}_rgb.jpg"
        Image.new("RGB", (32, 24), (frame_idx * 20, 30, 40)).save(run_dir / image_name)
        (run_dir / f"frame_{frame_idx:06d}_waypoint.json").write_text(
            __import__("json").dumps(
                    {
                        "frame_idx": frame_idx,
                        "rgb_file": image_name,
                        "response": {"discrete_action": [1] if frame_idx == 0 else [3]},
                    }
            )
        )
        result = upper_agent.evaluate_latest(run_dir, config_path, force=True)
        expected = library["commands"][frame_idx + 1]
        assert result["event"]["output"]["current_subgoal"] == expected
        runtime = load_runtime_config(config_path)
        assert runtime["instruction"] == expected
        assert runtime["upper_agent"]["hard_reset_requested"] is True
