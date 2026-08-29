#!/bin/zsh
set -u

umask 077

PROJECT_ROOT="/Users/clairehou/pyProjects/video_factory"
CLI="${PROJECT_ROOT}/.venv/bin/video-factory"
WORKSPACE="${PROJECT_ROOT}/workspace"
CONFIG="${PROJECT_ROOT}/examples/resource_discovery.json"
LOG_DIR="${WORKSPACE}/logs"

# A LaunchAgent does not inherit Terminal's environment. The user's shell
# profile already owns the provider credentials, so reuse it without copying
# secrets into this repository or the launchd plist.
if [[ -f "/Users/clairehou/.zshrc" ]]; then
  source "/Users/clairehou/.zshrc"
fi

export HOME="/Users/clairehou"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/clairehou/.local/bin:${PATH:-}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/discovery-$(date '+%Y-%m-%d').log"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] scheduled discovery started"
  if [[ ! -x "${CLI}" ]]; then
    echo "Video Factory CLI is missing or not executable: ${CLI}"
    exit_code=127
  else
    # No --force: Video Factory's persisted next_run_at controls each channel.
    # caffeinate keeps a long browser capture or render from being interrupted.
    /usr/bin/caffeinate -dimsu "${CLI}" \
      --workspace "${WORKSPACE}" discover \
      --config "${CONFIG}" \
      --provider deepseek
    exit_code=$?
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] scheduled discovery finished with status ${exit_code}"
  echo
} >>"${LOG_FILE}" 2>&1

# Keep enough history for diagnosis without allowing unattended logs to grow forever.
find "${LOG_DIR}" -type f -name 'discovery-*.log' -mtime +14 -delete 2>/dev/null || true
exit "${exit_code}"
