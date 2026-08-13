"""Cross-experiment memory for the real-world Upper Agent.

The per-run ``upper_agent_memory.json`` remains the authoritative working
memory. This module adds optional, best-effort long-term recall:

* Mem0 + local Qdrant for semantic retrieval across experiment runs.
* Raw ``infer=False`` writes, so memory storage never invokes another LLM.
* An append-only graph event stream that can later be imported into a graph DB.

All storage work is serialized on a background worker. A slow or unavailable
memory backend must never block the navigation control path.
"""

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "output" / "agent_memory"
DEFAULT_EMBEDDING_CACHE = PROJECT_ROOT / "checkpoints" / "embeddings"
DEFAULT_EMBEDDING_MODEL = str(DEFAULT_EMBEDDING_CACHE / "paraphrase-multilingual-MiniLM-L12-v2")
MEMORY_USER_ID = "internnav_robot"
MEMORY_AGENT_ID = "upper_agent"


def _clip(value, limit):
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _clean_list(values, limit=6):
    if not isinstance(values, list):
        values = [values] if values else []
    result = []
    for value in values:
        text = _clip(value, 60)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _grounded_findings(output):
    feedback = output.get("task_feedback") if isinstance(output.get("task_feedback"), dict) else {}
    grounded = []
    for item in feedback.get("findings") or []:
        if not isinstance(item, dict):
            continue
        description = _clip(item.get("description"), 120)
        evidence = _clip(item.get("evidence"), 140)
        if not description or not evidence:
            continue
        grounded.append(
            {
                "type": _clip(item.get("type") or "observation", 40),
                "description": description,
                "location": _clip(item.get("location"), 80),
                "severity": _clip(item.get("severity") or "info", 20),
                "evidence": evidence,
            }
        )
        if len(grounded) >= 5:
            break
    return grounded


def _normalize_result_text(output, completed, failure, findings):
    feedback = output.get("task_feedback") if isinstance(output.get("task_feedback"), dict) else {}
    status = str(output.get("task_status") or "running").strip().lower()
    if status == "completed":
        return completed or _clip(feedback.get("summary"), 100) or "task completed"
    if status == "failed":
        return failure or "task failed"
    if completed:
        return f"subgoal completed: {completed}"
    if failure:
        return f"failure: {failure}"
    if findings:
        return f"finding: {_clip(findings[0].get('description'), 90)}"
    return ""


def build_memory_candidate(
    previous_working_memory,
    output,
    *,
    run_name,
    frame_idx,
    task_instruction,
):
    """Return one compact high-value memory, or None for routine frame updates."""
    if not isinstance(output, dict):
        return None
    previous_working_memory = previous_working_memory or {}
    memory = output.get("memory") if isinstance(output.get("memory"), dict) else {}
    feedback = output.get("task_feedback") if isinstance(output.get("task_feedback"), dict) else {}
    previous_subgoal = _clip(previous_working_memory.get("active_subgoal"), 140)
    current_subgoal = _clip(output.get("current_subgoal"), 140)
    status = str(output.get("task_status") or "running").strip().lower()
    completed = _clip(memory.get("completed_subgoal"), 100)
    failure = _clip(memory.get("failure_reason") or feedback.get("failure_reason"), 100)
    findings = _grounded_findings(output)
    memory_types = []
    if failure or status == "failed":
        memory_types.append("failure")
    elif completed or status == "completed":
        memory_types.append("success")
    if findings:
        memory_types.append("finding")

    # A new command is working state, not durable experience. It becomes
    # long-term memory only after completion, grounded failure, or a finding
    # with explicit visual evidence.
    if not memory_types:
        return None

    place = _clip(memory.get("current_place") or previous_working_memory.get("current_place"), 90)
    landmarks = _clean_list(
        memory.get("landmarks_seen")
        or output.get("landmarks_seen")
        or previous_working_memory.get("visited_landmarks"),
        limit=5,
    )
    scene = _clip(memory.get("scene") or place or "unknown", 80)
    if "success" in memory_types:
        action = completed or previous_subgoal or current_subgoal
    elif "failure" in memory_types:
        action = previous_subgoal or current_subgoal
    else:
        action = current_subgoal or previous_subgoal
    result = _normalize_result_text(output, completed, failure, findings)
    task = _clip(task_instruction, 160)
    finding_text = "; ".join(
        f"{item['description']} (evidence: {item['evidence']})"
        for item in findings
    )
    parts = [
        f"memory_types={','.join(memory_types)}",
        f"scene={scene}",
        f"place={place}" if place else "",
        f"task={task}" if task else "",
        f"action={action}" if action else "",
        f"result={result}",
        f"findings={finding_text}" if finding_text else "",
        f"landmarks={','.join(landmarks)}" if landmarks else "",
    ]
    text = _clip(" | ".join(part for part in parts if part), 520)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "scene": scene,
                "place": place,
                "task": task,
                "action": action,
                "result": result,
                "landmarks": landmarks,
                "status": status,
                "memory_types": memory_types,
                "findings": findings,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "text": text,
        "memory_types": memory_types,
        "run_id": str(run_name or ""),
        "frame_idx": int(frame_idx or 0),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene": scene,
        "place": place,
        "task": task,
        "action": action,
        "result": result,
        "outcome": (
            "failed"
            if "failure" in memory_types
            else ("completed" if status == "completed" else ("subgoal_completed" if "success" in memory_types else "observed"))
        ),
        "task_status": status,
        "completed_subgoal": completed,
        "failure_reason": failure,
        "findings": findings,
        "landmarks": landmarks,
        "confidence": float(output.get("confidence") or 0.0),
    }


def build_retrieval_query(task_instruction, route_memory, current_instruction):
    route_memory = route_memory or {}
    parts = [
        _clip(task_instruction, 180),
        _clip(route_memory.get("current_place"), 100),
        _clip(current_instruction or route_memory.get("active_subgoal"), 120),
        ", ".join(_clean_list(route_memory.get("visited_landmarks"), limit=4)),
    ]
    return " | ".join(part for part in parts if part)


class JsonlGraphMemorySink:
    """Phase-two-ready graph event stream with no graph database dependency."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, candidate):
        memory_types = set(candidate.get("memory_types") or [])
        nodes = [
            {"type": "scene", "id": candidate["scene"]},
            {"type": "place", "id": candidate["place"]},
            {"type": "action", "id": candidate["action"]},
            {"type": "run", "id": candidate["run_id"]},
        ]
        nodes.extend({"type": "landmark", "id": item} for item in candidate["landmarks"])
        if candidate.get("failure_reason"):
            nodes.append({"type": "failure_reason", "id": candidate["failure_reason"]})
        nodes.extend(
            {"type": "finding", "id": item["description"]}
            for item in candidate.get("findings") or []
            if item.get("description")
        )
        nodes = [node for node in nodes if node["id"]]
        edges = []
        if candidate["scene"] and candidate["place"]:
            edges.append({"from": candidate["scene"], "relation": "CONTAINS", "to": candidate["place"]})
        if "success" in memory_types and candidate["action"] and candidate["place"]:
            edges.append({"from": candidate["action"], "relation": "SUCCEEDED_AT", "to": candidate["place"]})
        if "failure" in memory_types and candidate["action"] and candidate["place"]:
            edges.append({"from": candidate["action"], "relation": "FAILED_AT", "to": candidate["place"]})
        if "failure" in memory_types and candidate["action"] and candidate.get("failure_reason"):
            edges.append(
                {
                    "from": candidate["action"],
                    "relation": "FAILED_BECAUSE",
                    "to": candidate["failure_reason"],
                }
            )
        for landmark in candidate["landmarks"]:
            if candidate["place"]:
                edges.append({"from": candidate["place"], "relation": "HAS_LANDMARK", "to": landmark})
        for finding in candidate.get("findings") or []:
            if candidate["run_id"] and finding.get("description"):
                edges.append(
                    {
                        "from": candidate["run_id"],
                        "relation": "OBSERVED",
                        "to": finding["description"],
                    }
                )
        record = {
            "schema_version": 2,
            "event_type": "navigation_memory",
            "memory_types": sorted(memory_types),
            "fingerprint": candidate["fingerprint"],
            "run_id": candidate["run_id"],
            "frame_idx": candidate["frame_idx"],
            "created_at": candidate["created_at"],
            "nodes": nodes,
            "edges": edges,
            "source": candidate,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class LongTermMemoryManager:
    def __init__(self, settings):
        self.settings = dict(settings or {})
        self.root = Path(self.settings.get("root") or DEFAULT_MEMORY_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.graph_sink = JsonlGraphMemorySink(self.root / "graph_events.jsonl")
        self._qdrant_path = self.root / "qdrant"
        self._qdrant_lock_path = self.root / ".qdrant.lock"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="internnav-memory")
        self._memory = None
        self._state = "idle"
        self._error = ""
        self._lock = threading.Lock()
        self._seen = self._load_recent_fingerprints()
        self._pending = 0
        self._last_search_ms = 0.0
        self._last_write_at = ""
        self._executor.submit(self._initialize)

    def _load_recent_fingerprints(self):
        path = self.root / "graph_events.jsonl"
        if not path.exists():
            return set()
        fingerprints = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                        if value.get("fingerprint"):
                            fingerprints.append(value["fingerprint"])
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            return set()
        return set(fingerprints[-5000:])

    def _initialize(self):
        with self._lock:
            self._state = "loading"
            self._error = ""
        try:
            os.environ["MEM0_DIR"] = str(self.root / "mem0")
            os.environ["MEM0_TELEMETRY"] = "False"
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from mem0 import Memory

            embedding_model = str(self.settings.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
            embedding_cache = Path(self.settings.get("embedding_cache") or DEFAULT_EMBEDDING_CACHE)
            embedding_cache.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "v1.1",
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "internnav_long_term_memory",
                        "path": str(self.root / "qdrant"),
                        "embedding_model_dims": 384,
                        "on_disk": True,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": embedding_model,
                        "embedding_dims": 384,
                        "model_kwargs": {
                            "cache_folder": str(embedding_cache),
                            "device": "cpu",
                            "local_files_only": True,
                        },
                    },
                },
                # infer=False is used for every write. This client is created
                # only because Mem0 requires an LLM provider in its config.
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "unused-memory-extractor",
                        "api_key": "local-memory-no-llm",
                        "openai_base_url": "http://127.0.0.1:1/v1",
                    },
                },
                "history_db_path": str(self.root / "history.db"),
            }
            from filelock import FileLock

            with FileLock(str(self._qdrant_lock_path), timeout=20):
                self._memory = Memory.from_config(config)
                self._close_vector_clients()
            with self._lock:
                self._state = "ready"
        except Exception as exc:
            with self._lock:
                self._state = "degraded"
                self._error = f"{type(exc).__name__}: {exc}"
            self._memory = None

    def _close_vector_clients(self):
        for attribute in ("vector_store", "_telemetry_vector_store"):
            store = getattr(self._memory, attribute, None)
            client = getattr(store, "client", None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
                store.client = None

    @contextmanager
    def _vector_session(self, timeout):
        """Serialize short Qdrant sessions across Server and Viewer processes."""
        from filelock import FileLock
        from qdrant_client import QdrantClient

        with FileLock(str(self._qdrant_lock_path), timeout=max(0.02, float(timeout))):
            store = self._memory.vector_store
            store.client = QdrantClient(path=str(self._qdrant_path))
            try:
                yield
            finally:
                self._close_vector_clients()

    def _search(self, query, top_k, char_budget):
        if self._memory is None:
            return []
        started = time.perf_counter()
        with self._vector_session(timeout=0.12):
            response = self._memory.search(
                query,
                user_id=MEMORY_USER_ID,
                agent_id=MEMORY_AGENT_ID,
                limit=top_k,
                rerank=False,
            )
        results = response.get("results", []) if isinstance(response, dict) else response
        compact = []
        remaining = max(0, int(char_budget))
        for item in results or []:
            raw_text = str(item.get("memory") if isinstance(item, dict) else item or "").strip()
            prefix = raw_text.split("|", 1)[0].strip()
            if not prefix.startswith("memory_types="):
                # Legacy entries can contain unverified command transitions.
                # Preserve them on disk, but never return them to the agent.
                continue
            memory_types = [
                value.strip()
                for value in prefix.removeprefix("memory_types=").split(",")
                if value.strip() in {"success", "failure", "finding"}
            ]
            if not memory_types:
                continue
            text = _clip(raw_text, min(220, remaining))
            if not text or remaining <= 0:
                break
            compact.append(
                {
                    "memory": text,
                    "memory_types": memory_types,
                    "score": round(float((item or {}).get("score") or 0.0), 3)
                    if isinstance(item, dict)
                    else 0.0,
                }
            )
            remaining -= len(text)
        self._last_search_ms = (time.perf_counter() - started) * 1000.0
        return compact

    def retrieve(self, query, *, top_k=3, char_budget=600, timeout_ms=180):
        if not query or self._state != "ready":
            return []
        future = self._executor.submit(self._search, query, int(top_k), int(char_budget))
        try:
            return future.result(timeout=max(0.01, float(timeout_ms) / 1000.0))
        except TimeoutError:
            return []
        except Exception as exc:
            with self._lock:
                self._error = f"search {type(exc).__name__}: {exc}"
            return []

    def _write(self, candidate, capture_graph):
        try:
            if capture_graph:
                self.graph_sink.append(candidate)
            if self._memory is not None:
                metadata = {
                    key: candidate[key]
                    for key in (
                        "fingerprint",
                        "run_id",
                        "frame_idx",
                        "scene",
                        "place",
                        "task",
                        "action",
                        "result",
                        "outcome",
                        "confidence",
                    )
                }
                metadata["landmarks"] = ",".join(candidate["landmarks"])
                metadata["memory_types"] = ",".join(candidate["memory_types"])
                metadata["completed_subgoal"] = candidate["completed_subgoal"]
                metadata["failure_reason"] = candidate["failure_reason"]
                with self._vector_session(timeout=5.0):
                    self._memory.add(
                        candidate["text"],
                        user_id=MEMORY_USER_ID,
                        agent_id=MEMORY_AGENT_ID,
                        metadata=metadata,
                        infer=False,
                    )
            self._last_write_at = candidate["created_at"]
        except Exception as exc:
            with self._lock:
                self._error = f"write {type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._pending = max(0, self._pending - 1)

    def remember_async(self, candidate, *, capture_graph=True):
        if not candidate:
            return False
        fingerprint = candidate.get("fingerprint")
        with self._lock:
            if not fingerprint or fingerprint in self._seen or self._pending >= 64:
                return False
            self._seen.add(fingerprint)
            self._pending += 1
        self._executor.submit(self._write, candidate, bool(capture_graph))
        return True

    def status(self):
        with self._lock:
            return {
                "state": self._state,
                "backend": "mem0-qdrant-local",
                "embedding_device": "cpu",
                "pending_writes": self._pending,
                "last_search_ms": round(self._last_search_ms, 2),
                "last_write_at": self._last_write_at,
                "graph_event_path": str(self.root / "graph_events.jsonl"),
                "error": self._error,
            }


_MANAGER = None
_MANAGER_LOCK = threading.Lock()


def get_long_term_memory_manager(settings=None):
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = LongTermMemoryManager(settings or {})
        return _MANAGER


def long_term_memory_status():
    if _MANAGER is None:
        return {
            "state": "idle",
            "backend": "mem0-qdrant-local",
            "embedding_device": "cpu",
            "pending_writes": 0,
            "last_search_ms": 0.0,
            "last_write_at": "",
            "graph_event_path": str(DEFAULT_MEMORY_ROOT / "graph_events.jsonl"),
            "error": "",
        }
    return _MANAGER.status()
