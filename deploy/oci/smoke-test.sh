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
MPT_ROOT="${MPT_ROOT:-/opt/video-factory/MoneyPrinterTurbo}"
WEB_ROOT="${WEB_SCROLL_VIDEO_ROOT:-/opt/video-factory/web-scroll-video}"
CHROME="${VIDEO_FACTORY_CHROME_PATH:-/opt/video-factory/bin/chromium}"
FONT="${VIDEO_FACTORY_FONT:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}"
BGM="${VIDEO_FACTORY_BGM_FILE:-}"

failures=0
check_command() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    echo "ok command ${name}: $(command -v "${name}")"
  else
    echo "FAIL missing command: ${name}" >&2
    failures=$((failures + 1))
  fi
}
check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "ok ${label}: ${path}"
  else
    echo "FAIL missing ${label}: ${path}" >&2
    failures=$((failures + 1))
  fi
}

arch="$(uname -m)"
if [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then
  echo "ok architecture: ${arch}"
else
  echo "FAIL expected Oracle A1 ARM64, found ${arch}" >&2
  failures=$((failures + 1))
fi

for command in ffmpeg ffprobe flock git node deno sha256sum uv sqlite3; do
  check_command "${command}"
done

node_major="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/' || echo 0)"
if [[ "${node_major}" =~ ^[0-9]+$ ]] && (( node_major >= 22 )); then
  echo "ok Node major: ${node_major}"
else
  echo "FAIL Node 22+ required, found $(node --version 2>/dev/null || echo missing)" >&2
  failures=$((failures + 1))
fi

check_file "Video Factory CLI" "${CLI}"
check_file "MoneyPrinterTurbo CLI" "${MPT_ROOT}/cli.py"
check_file "web-scroll-video runner" "${WEB_ROOT}/src/scroll-video.mjs"
check_file "Chinese font" "${FONT}"
check_file "managed Chromium" "${CHROME}"
if [[ -n "${BGM}" ]]; then
  check_file "licensed background music" "${BGM}"
fi

mkdir -p "${WORKSPACE}"
if [[ -w "${WORKSPACE}" ]]; then
  echo "ok writable workspace: ${WORKSPACE}"
else
  echo "FAIL workspace is not writable: ${WORKSPACE}" >&2
  failures=$((failures + 1))
fi

ffmpeg_encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
if ! grep -q 'libx264' <<<"${ffmpeg_encoders}"; then
  echo "FAIL ffmpeg lacks libx264 encoder" >&2
  failures=$((failures + 1))
else
  echo "ok ffmpeg libx264 encoder"
fi

if [[ -x "${CHROME}" ]]; then
  if "${CHROME}" --headless --no-sandbox --disable-gpu --dump-dom about:blank \
    >/dev/null 2>&1; then
    echo "ok managed Chromium headless launch"
  else
    echo "FAIL managed Chromium could not launch headlessly" >&2
    failures=$((failures + 1))
  fi
fi

if [[ -x "${CLI}" ]]; then
  if "${CLI}" --workspace "${WORKSPACE}" publish-policy >/dev/null; then
    echo "ok Video Factory CLI"
  else
    echo "FAIL Video Factory CLI policy check" >&2
    failures=$((failures + 1))
  fi
fi

if [[ -n "${VIDEO_FACTORY_SMOKE_PUBLISH_PLATFORM:-}" ]]; then
  account="${VIDEO_FACTORY_SMOKE_PUBLISH_ACCOUNT:-main}"
  if "${CLI}" --workspace "${WORKSPACE}" publisher check \
    "${VIDEO_FACTORY_SMOKE_PUBLISH_PLATFORM}" --account "${account}"; then
    echo "ok publisher login: ${VIDEO_FACTORY_SMOKE_PUBLISH_PLATFORM}/${account}"
  else
    echo "FAIL publisher login: ${VIDEO_FACTORY_SMOKE_PUBLISH_PLATFORM}/${account}" >&2
    failures=$((failures + 1))
  fi
fi

if (( failures > 0 )); then
  echo "smoke test failed with ${failures} problem(s)" >&2
  exit 1
fi
echo "all Oracle deployment smoke checks passed"
