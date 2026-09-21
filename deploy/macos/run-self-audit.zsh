#!/bin/zsh
set -u

umask 077

PROJECT_ROOT="/Users/clairehou/pyProjects/video_factory"
CLI="${PROJECT_ROOT}/.venv/bin/video-factory"
WORKSPACE="${PROJECT_ROOT}/workspace"
LOG_DIR="${WORKSPACE}/logs"

# LaunchAgents do not inherit the interactive shell's provider credentials.
# Reuse the user's shell configuration without copying secrets into the plist.
if [[ -f "/Users/clairehou/.zshrc" ]]; then
  source "/Users/clairehou/.zshrc"
fi

export HOME="/Users/clairehou"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/clairehou/.local/bin:${PATH:-}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/self-audit-$(date '+%Y-%m-%d').log"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] nightly self-audit started"
  if [[ ! -x "${CLI}" ]]; then
    echo "Video Factory CLI is missing or not executable: ${CLI}"
    exit_code=127
  else
    /usr/bin/caffeinate -dimsu "${CLI}" \
      --workspace "${WORKSPACE}" self-audit run \
      --max-issues 5 \
      --max-cost-usd 1.0
    exit_code=$?
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] nightly self-audit finished with status ${exit_code}"
  echo
} >>"${LOG_FILE}" 2>&1

find "${LOG_DIR}" -type f -name 'self-audit-*.log' -mtime +30 -delete 2>/dev/null || true
exit "${exit_code}"
