#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 <tencent|douyin|xiaohongshu|bilibili> [account]" >&2
  exit 2
fi

ENV_FILE="${VIDEO_FACTORY_ENV_FILE:-/etc/video-factory/video-factory.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PLATFORM="$1"
ACCOUNT="${2:-main}"
CLI="${VIDEO_FACTORY_CLI:-/opt/video-factory/venv/bin/video-factory}"
WORKSPACE="${VIDEO_FACTORY_WORKSPACE:-/srv/video-factory/workspace}"
DISPLAY_NUMBER="${VIDEO_FACTORY_LOGIN_DISPLAY:-99}"
VNC_PORT="${VIDEO_FACTORY_LOGIN_VNC_PORT:-5900}"
RUNTIME="${VIDEO_FACTORY_RUNTIME:-/srv/video-factory/runtime}"
SESSION_DIR="${RUNTIME}/login-desktop"

mkdir -p "${SESSION_DIR}"
cleanup() {
  stop_process() {
    local pid_file="$1"
    local expected="$2"
    if [[ -f "${pid_file}" ]]; then
      pid="$(cat "${pid_file}")"
      process="$(ps -p "${pid}" -o comm= 2>/dev/null | tr -d '[:space:]' || true)"
      if [[ "${process}" == *"${expected}"* ]]; then
        kill "${pid}" 2>/dev/null || true
      fi
      rm -f "${pid_file}"
    fi
  }
  stop_process "${SESSION_DIR}/x11vnc.pid" "x11vnc"
  stop_process "${SESSION_DIR}/fluxbox.pid" "fluxbox"
  stop_process "${SESSION_DIR}/xvfb.pid" "Xvfb"
}
trap cleanup EXIT INT TERM
cleanup

Xvfb ":${DISPLAY_NUMBER}" -screen 0 1440x900x24 -nolisten tcp \
  >"${SESSION_DIR}/xvfb.log" 2>&1 &
echo "$!" >"${SESSION_DIR}/xvfb.pid"
export DISPLAY=":${DISPLAY_NUMBER}"
sleep 1

fluxbox >"${SESSION_DIR}/fluxbox.log" 2>&1 &
echo "$!" >"${SESSION_DIR}/fluxbox.pid"
x11vnc -display "${DISPLAY}" -localhost -rfbport "${VNC_PORT}" \
  -forever -shared -nopw >"${SESSION_DIR}/x11vnc.log" 2>&1 &
echo "$!" >"${SESSION_DIR}/x11vnc.pid"

echo "VNC is listening only on server localhost:${VNC_PORT}."
echo "From your Mac, open another terminal and run:"
echo "  ssh -N -L 5901:127.0.0.1:${VNC_PORT} <oracle-user>@<oracle-ip>"
echo "Then open: vnc://127.0.0.1:5901"
echo
echo "Waiting five seconds before opening the ${PLATFORM} login window..."
sleep 5

"${CLI}" --workspace "${WORKSPACE}" publisher login \
  "${PLATFORM}" --account "${ACCOUNT}"
