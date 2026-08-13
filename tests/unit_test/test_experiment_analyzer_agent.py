import json
import sys
from pathlib import Path

from PIL import Image


REALWORLD_DIR = Path(__file__).resolve().parents[2] / "scripts" / "realworld"
if str(REALWORLD_DIR) not in sys.path:
    sys.path.insert(0, str(REALWORLD_DIR))

from experiment_analyzer_agent import (  # noqa: E402
    _coarse_visual_retrieve,
    build_experiment_index,
    index_status,
    retrieve_evidence,
)
from experiment_instance_analyzer import (  # noqa: E402
    DETECTIONS_FILENAME,
    build_instance_index,
    instance_evidence,
)
from yolo_instance_worker import _resolve_class_ids  # noqa: E402
import experiment_visualizer  # noqa: E402
from runtime_config import save_runtime_config  # noqa: E402


def make_run(root, name="20260720_100000"):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    actions = [1, 1, 2, 0, 0, 1]
    instructions = ["Go straight.", "Go straight.", "Turn left.", "Turn left.", "Turn left.", "Continue straight."]
    colors = [(20, 30, 40), (22, 32, 42), (150, 40, 30), (152, 42, 32), (154, 44, 34), (20, 130, 80)]
    for frame_idx, (action, instruction, color) in enumerate(zip(actions, instructions, colors)):
        rgb_file = f"frame_{frame_idx:06d}_rgb.jpg"
        depth_file = f"frame_{frame_idx:06d}_depth.png"
        vis_file = f"frame_{frame_idx:06d}_vis.jpg"
        Image.new("RGB", (64, 48), color).save(run_dir / rgb_file)
        Image.new("I;16", (64, 48), 1000).save(run_dir / depth_file)
        Image.new("RGB", (64, 48), color).save(run_dir / vis_file)
        (run_dir / f"frame_{frame_idx:06d}_waypoint.json").write_text(
            json.dumps(
                {
                    "frame_idx": frame_idx,
                    "saved_at": f"2026-07-20T10:00:{frame_idx:02d}",
                    "instruction": instruction,
                    "response": {"discrete_action": [action]},
                    "rgb_file": rgb_file,
                    "depth_file": depth_file,
                    "vis_file": vis_file,
                }
            )
        )
    return run_dir


def test_index_preserves_action_changes_and_stop_sequence(tmp_path):
    run_dir = make_run(tmp_path)
    index = build_experiment_index(run_dir, max_keyframes=20, sample_interval=50)

    assert index["source"]["frame_count"] == 6
    assert index["stop_events"] == [{"start_frame": 3, "end_frame": 4, "saved_frame_count": 2}]
    keyframe_indexes = {item["frame_idx"] for item in index["keyframes"]}
    assert {0, 2, 3, 4, 5}.issubset(keyframe_indexes)
    assert index_status(run_dir)["stale"] is False


def test_stop_question_retrieves_stop_evidence(tmp_path):
    index = build_experiment_index(make_run(tmp_path), max_keyframes=20, sample_interval=50)
    evidence = retrieve_evidence(index, "机器人为什么出现异常停顿？", limit=3)

    assert any(item["action"]["is_stop"] for item in evidence)
    assert any(item["frame_idx"] in {3, 4} for item in evidence)


def test_viewer_analysis_index_and_history_api(tmp_path):
    log_root = tmp_path / "runs"
    run_dir = make_run(log_root)
    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": False, "upper_agent": {}})
    app = experiment_visualizer.create_viewer_app(log_root, config_path)
    app.testing = True
    client = app.test_client()

    response = client.post(f"/api/experiment-analysis/{run_dir.name}/index", json={"force": True})
    assert response.status_code == 200
    assert response.get_json()["summary"]["keyframe_count"] >= 2

    response = client.get(f"/api/experiment-analysis/{run_dir.name}/history")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "records": []}

    (run_dir / DETECTIONS_FILENAME).write_text(
        json.dumps(
            {
                "frame_idx": 2,
                "detections": [
                    {"label": "couch", "score": 0.8, "bbox_xyxy": [10, 10, 40, 40]}
                ],
            }
        )
    )
    (run_dir / "experiment_instance_detector_meta.json").write_text(
        json.dumps({"processed_frame_count": 6, "all_classes": True})
    )
    response = client.post(
        f"/api/experiment-analysis/{run_dir.name}/instances/index",
        json={"run_detector": True, "minimum_score": 0.1},
    )
    assert response.status_code == 200
    assert response.get_json()["detector_ran"] is False
    assert response.get_json()["instance_count"] == 1

    page = client.get(f"/run/{run_dir.name}")
    assert page.status_code == 200
    assert b"Experiment QA" in page.data


def test_contact_sheet_retrieval_uses_labeled_candidate_frames(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    index = build_experiment_index(run_dir, max_keyframes=20, sample_interval=1)

    class FakeResponse:
        ok = True

        def json(self):
            return {"choices": [{"message": {"content": '{"candidate_frames":[2,4],"reason":"sofa candidates"}'}}]}

    monkeypatch.setattr("experiment_analyzer_agent.requests.post", lambda *args, **kwargs: FakeResponse())
    candidates, _ = _coarse_visual_retrieve(
        {"api_key": "server-only", "api_url": "https://example.invalid", "model": "vision-model"},
        "有没有灰色沙发？",
        index,
        run_dir,
    )
    assert [item["frame_idx"] for item in candidates] == [2, 4]


def test_home_qa_forwards_selected_model_without_exposing_key(tmp_path, monkeypatch):
    log_root = tmp_path / "runs"
    run_dir = make_run(log_root)
    config_path = tmp_path / "runtime.json"
    save_runtime_config(
        config_path,
        {"service_enabled": False, "upper_agent": {"api_key": "secret", "model": "default-model"}},
    )
    captured = {}

    def fake_answer(run_dir_arg, question, config, max_images=8):
        captured.update({"run_dir": Path(run_dir_arg), "question": question, "model": config["model"]})
        return {"question": question, "answer": "found", "confidence": 0.9, "evidence": [], "model": config["model"]}

    monkeypatch.setattr(experiment_visualizer, "answer_experiment_question", fake_answer)
    monkeypatch.setattr(
        experiment_visualizer,
        "ensure_full_frame_instance_index",
        lambda *args, **kwargs: {"detector_ran": False, "index": {}},
    )
    app = experiment_visualizer.create_viewer_app(log_root, config_path)
    app.testing = True
    client = app.test_client()
    response = client.post(
        f"/api/experiment-analysis/{run_dir.name}/ask",
        json={"question": "有没有灰色沙发？", "model": "selected-vision-model"},
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert captured["model"] == "selected-vision-model"
    assert "secret" not in json.dumps(payload)
    home = client.get("/")
    assert b"homeQaQuestion" in home.data
    assert b"selected-vision-model" not in home.data


def test_full_frame_tracks_merge_with_rgbd_world_evidence(tmp_path):
    run_dir = tmp_path / "20260720_120000"
    run_dir.mkdir()
    intrinsic = [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]
    world_t_camera = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for frame_idx in (1, 20):
        rgb_file = f"frame_{frame_idx:06d}_rgb.jpg"
        depth_file = f"frame_{frame_idx:06d}_depth.png"
        Image.new("RGB", (64, 48), (100, 100, 100)).save(run_dir / rgb_file)
        Image.new("I;16", (64, 48), 1000).save(run_dir / depth_file)
        (run_dir / f"frame_{frame_idx:06d}_waypoint.json").write_text(
            json.dumps(
                {
                    "frame_idx": frame_idx,
                    "rgb_file": rgb_file,
                    "depth_file": depth_file,
                    "camera_intrinsic": intrinsic,
                    "world_T_camera": world_t_camera,
                }
            )
        )
    detections = [
        {"frame_idx": 1, "detections": [
            {"track_id": "a", "label": "sofa", "score": 0.8, "bbox_xyxy": [20, 12, 44, 36], "mask_centroid": [32, 24]},
            {"track_id": "nearby", "label": "sofa", "score": 0.7, "bbox_xyxy": [2, 12, 18, 36], "mask_centroid": [10, 24]},
        ]},
        {"frame_idx": 20, "detections": [{"track_id": "b", "label": "couch", "score": 0.9, "bbox_xyxy": [20, 12, 44, 36], "mask_centroid": [32, 24]}]},
    ]
    (run_dir / DETECTIONS_FILENAME).write_text("\n".join(json.dumps(item) for item in detections))

    index = build_instance_index(run_dir, max_frame_gap=2)

    assert index["capabilities"]["cross_loop_unique_count"] is True
    assert len(index["instances"]) == 1
    assert index["detection_summary"]["sofa"]["max_simultaneous"] == 2
    merged = max(index["instances"], key=lambda item: item["visible_frame_count"])
    assert merged["visible_frame_count"] == 3
    evidence, instances = instance_evidence(index, "一共有几个灰色沙发？")
    assert len(instances) == 1
    assert {item["frame_idx"] for item in evidence} == {1, 20}


def test_yolo_worker_resolves_requested_full_frame_classes():
    names = {0: "person", 56: "chair", 57: "couch"}

    assert _resolve_class_ids(names, ["couch", "chair"]) == [56, 57]


def test_visual_absence_is_not_high_confidence_without_full_frame_index(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(
        "experiment_analyzer_agent._coarse_visual_retrieve",
        lambda *args, **kwargs: ([], 0.0),
    )
    monkeypatch.setattr(
        "experiment_analyzer_agent._call_vlm",
        lambda *args, **kwargs: (
            {
                "answer": "没有看到灰色沙发。",
                "confidence": 0.95,
                "uncertainty": "",
                "time_ranges": [],
                "evidence": [],
                "data_sources": ["RGB keyframes"],
            },
            0.01,
        ),
    )
    from experiment_analyzer_agent import answer_question

    record = answer_question(run_dir, "有没有灰色沙发？", {"model": "fake"})

    assert record["instance_index_available"] is False
    assert record["confidence"] == 0.65
    assert "Full-frame detector/tracker index is unavailable" in record["uncertainty"]


def test_visual_qa_prefers_full_frame_instance_index(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    (run_dir / DETECTIONS_FILENAME).write_text(
        json.dumps(
            {
                "frame_idx": 2,
                "detections": [
                    {"label": "couch", "score": 0.85, "bbox_xyxy": [8, 8, 56, 42]}
                ],
            }
        )
    )
    build_instance_index(run_dir, minimum_score=0.1)
    monkeypatch.setattr(
        "experiment_analyzer_agent.build_experiment_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full-frame evidence must not rebuild the thumbnail index")
        ),
    )
    monkeypatch.setattr(
        "experiment_analyzer_agent._call_vlm",
        lambda *args, **kwargs: (
            {
                "answer": "候选帧中检测到灰色沙发。",
                "confidence": 0.9,
                "uncertainty": "",
                "time_ranges": [],
                "evidence": [{"frame_idx": 2, "reason": "Full-frame candidate."}],
                "data_sources": ["full-frame instance index"],
            },
            0.01,
        ),
    )
    from experiment_analyzer_agent import answer_question

    record = answer_question(run_dir, "有没有灰色沙发？", {"model": "fake"})

    assert record["instance_index_available"] is True
    assert record["search_strategy"] == "full_frame_instance_index_then_vlm"
    assert 2 in record["retrieved_frame_indexes"]


def test_experiment_analysis_progress_api(tmp_path):
    log_root = tmp_path / "runs"
    run_dir = make_run(log_root)
    config_path = tmp_path / "runtime.json"
    save_runtime_config(config_path, {"service_enabled": False})
    experiment_visualizer.write_analysis_progress(
        run_dir, "detecting", 42, "全帧目标检测 96/240", 96, 240
    )

    app = experiment_visualizer.create_viewer_app(log_root, config_path)
    app.testing = True
    response = app.test_client().get(
        f"/api/experiment-analysis/{run_dir.name}/progress"
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["progress"]["phase"] == "detecting"
    assert payload["progress"]["percent"] == 42
    assert payload["progress"]["current"] == 96
