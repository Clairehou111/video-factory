#!/bin/zsh
set -u

umask 077

PROJECT_ROOT="/Users/clairehou/pyProjects/video_factory"
CLI="${PROJECT_ROOT}/.venv/bin/video-factory"
WORKSPACE="${PROJECT_ROOT}/workspace"

if [[ -f "/Users/clairehou/.zshrc" ]]; then
  source "/Users/clairehou/.zshrc"
fi

export HOME="/Users/clairehou"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/clairehou/.local/bin:${PATH:-}"

if [[ ! -x "${CLI}" ]]; then
  echo "Video Factory CLI is missing or not executable: ${CLI}"
  exit 127
fi

exec "${CLI}" --workspace "${WORKSPACE}" dashboard \
  --host 127.0.0.1 --port 8765 --actor claire
