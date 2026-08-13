#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${INTERNNAV_PYTHON:-python}"
HOST="${INTERNNAV_VIEWER_HOST:-0.0.0.0}"
PORT="${INTERNNAV_VIEWER_PORT:-8899}"
LOG_DIR="${INTERNNAV_LOG_DIR:-${ROOT_DIR}/output/realworld_experiments}"
RUNTIME_CONFIG="${INTERNNAV_RUNTIME_CONFIG:-${ROOT_DIR}/output/realworld_runtime_config.json}"
TUNNEL_USER="${INTERNNAV_TUNNEL_USER:-${USER:-user}}"
TUNNEL_HOST="${INTERNNAV_TUNNEL_HOST:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Set INTERNNAV_PYTHON to the intended environment's Python path." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname "${RUNTIME_CONFIG}")"
cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" scripts/realworld/experiment_visualizer.py \
  --serve \
  --host "${HOST}" \
  --port "${PORT}" \
  --log_dir "${LOG_DIR}" \
  --runtime_config_path "${RUNTIME_CONFIG}" \
  --tunnel_user "${TUNNEL_USER}" \
  --tunnel_host "${TUNNEL_HOST}"
