#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE="${VIDEO_FACTORY_ENV_FILE:-/etc/video-factory/video-factory.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CLI="${VIDEO_FACTORY_CLI:-/opt/video-factory/venv/bin/video-factory}"
WORKSPACE="${VIDEO_FACTORY_WORKSPACE:-/srv/video-factory/workspace}"
CONFIG="${VIDEO_FACTORY_DISCOVERY_CONFIG:-/opt/video-factory/app/examples/resource_discovery.json}"
PROVIDER="${VIDEO_FACTORY_DISCOVERY_PROVIDER:-auto}"
CHANNEL_LIST="${VIDEO_FACTORY_DISCOVERY_CHANNELS:-github,official,paper,news,openrouter}"
LOCK_DIR="${VIDEO_FACTORY_RUNTIME:-/srv/video-factory/runtime}"

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for single-worker discovery execution" >&2
  exit 1
fi
mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_DIR}/discovery.lock"
if ! flock -n 9; then
  echo "another discovery run is active; skipping this timer tick"
  exit 0
fi

args=(--workspace "${WORKSPACE}" discover --config "${CONFIG}" --provider "${PROVIDER}")
IFS=',' read -r -a channels <<< "${CHANNEL_LIST}"
for channel in "${channels[@]}"; do
  channel="${channel//[[:space:]]/}"
  if [[ -n "${channel}" ]]; then
    args+=(--channel "${channel}")
  fi
done

exec "${CLI}" "${args[@]}"
