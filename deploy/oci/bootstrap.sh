#!/usr/bin/env bash
set -Eeuo pipefail

MPT_REPOSITORY="https://github.com/harry0703/MoneyPrinterTurbo.git"
MPT_COMMIT="d4c0e45da4ac0889af77f7307f52f9d5d4f74942"
WEB_SCROLL_REPOSITORY="https://github.com/upenn/web-scroll-video.git"
WEB_SCROLL_COMMIT="7c004aefb8ec4610a18ad21577105a9ddce60b15"
UV_VERSION="0.11.3"
DENO_VERSION="v2.8.3"

APP_USER="video-factory"
APP_ROOT="/opt/video-factory"
STATE_ROOT="/srv/video-factory"
CONFIG_ROOT="/etc/video-factory"
SOURCE_DIR="${1:-$(pwd)}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "bootstrap must run as root (use sudo)" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_DIR}/pyproject.toml" || ! -d "${SOURCE_DIR}/src/video_factory" ]]; then
  echo "source directory is not the Video Factory project: ${SOURCE_DIR}" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap supports Ubuntu Linux only" >&2
  exit 1
fi
if [[ ! -f /etc/os-release ]]; then
  echo "cannot identify the operating system" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "bootstrap requires Ubuntu 24.04; found ${PRETTY_NAME:-unknown}" >&2
  exit 1
fi
case "$(uname -m)" in
  aarch64|arm64) ;;
  *)
    echo "bootstrap requires an Oracle Ampere A1 ARM64 instance" >&2
    exit 1
    ;;
esac

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl ffmpeg fluxbox fonts-liberation \
  fonts-noto-cjk git jq libasound2t64 libatk-bridge2.0-0 libatk1.0-0 \
  libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 \
  libnspr4 libnss3 libpango-1.0-0 libx11-6 libatspi2.0-0 libxcb1 \
  libxcomposite1 libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2 \
  pipx rsync sqlite3 \
  unzip util-linux x11vnc xvfb

if ! id "${APP_USER}" >/dev/null 2>&1; then
  useradd --create-home --home-dir "${STATE_ROOT}" --shell /bin/bash "${APP_USER}"
fi

install -d -o "${APP_USER}" -g "${APP_USER}" -m 0750 \
  "${APP_ROOT}" "${APP_ROOT}/app" "${APP_ROOT}/bin" \
  "${STATE_ROOT}" "${STATE_ROOT}/workspace" "${STATE_ROOT}/runtime" \
  "${STATE_ROOT}/backups"
install -d -o root -g "${APP_USER}" -m 0750 "${CONFIG_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  uv_installer="$(mktemp)"
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${uv_installer}"
  UV_INSTALL_DIR=/usr/local/bin sh "${uv_installer}"
  rm -f "${uv_installer}"
fi

node_major=0
if command -v node >/dev/null 2>&1; then
  node_major="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
fi
if (( node_major < 22 )); then
  nodesource_installer="$(mktemp)"
  curl -fsSL https://deb.nodesource.com/setup_22.x -o "${nodesource_installer}"
  bash "${nodesource_installer}"
  rm -f "${nodesource_installer}"
  apt-get install -y --no-install-recommends nodejs
fi

if ! command -v deno >/dev/null 2>&1; then
  deno_installer="$(mktemp)"
  curl -fsSL https://deno.land/install.sh -o "${deno_installer}"
  DENO_INSTALL="${APP_ROOT}/deno" sh "${deno_installer}" "${DENO_VERSION}"
  rm -f "${deno_installer}"
  ln -sfn "${APP_ROOT}/deno/bin/deno" /usr/local/bin/deno
fi

source_real="$(realpath "${SOURCE_DIR}")"
app_real="$(realpath "${APP_ROOT}/app")"
if [[ "${source_real}" != "${app_real}" ]]; then
  rsync -a \
    --exclude '.git/' --exclude '.venv/' --exclude 'workspace/' \
    --exclude '__pycache__/' --exclude '*.pyc' \
    "${SOURCE_DIR}/" "${APP_ROOT}/app/"
fi
chown -R "${APP_USER}:${APP_USER}" "${APP_ROOT}/app"

clone_pinned() {
  local repository="$1"
  local commit="$2"
  local target="$3"
  if [[ -e "${target}" && ! -d "${target}/.git" ]]; then
    echo "refusing to replace a non-git dependency path: ${target}" >&2
    exit 1
  fi
  if [[ ! -d "${target}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${repository}" "${target}"
  fi
  git -C "${target}" fetch --depth 1 origin "${commit}"
  git -C "${target}" checkout --detach "${commit}"
}

clone_pinned "${MPT_REPOSITORY}" "${MPT_COMMIT}" "${APP_ROOT}/MoneyPrinterTurbo"
clone_pinned "${WEB_SCROLL_REPOSITORY}" "${WEB_SCROLL_COMMIT}" "${APP_ROOT}/web-scroll-video"
chown -R "${APP_USER}:${APP_USER}" \
  "${APP_ROOT}/MoneyPrinterTurbo" "${APP_ROOT}/web-scroll-video"

runuser -u "${APP_USER}" -- env HOME="${STATE_ROOT}" uv python install 3.11
runuser -u "${APP_USER}" -- env HOME="${STATE_ROOT}" uv venv --python 3.11 "${APP_ROOT}/venv"
runuser -u "${APP_USER}" -- env HOME="${STATE_ROOT}" uv pip install \
  --python "${APP_ROOT}/venv/bin/python" -e "${APP_ROOT}/app"
runuser -u "${APP_USER}" -- env HOME="${STATE_ROOT}" bash -lc \
  "cd '${APP_ROOT}/MoneyPrinterTurbo' && uv sync --frozen --python 3.11"

env_target="${CONFIG_ROOT}/video-factory.env"
if [[ ! -f "${env_target}" ]]; then
  install -o root -g "${APP_USER}" -m 0640 \
    "${APP_ROOT}/app/deploy/oci/video-factory.env.example" "${env_target}"
fi

set -a
# shellcheck disable=SC1090
source "${env_target}"
set +a

runuser -u "${APP_USER}" -- env \
  HOME="${STATE_ROOT}" \
  PATH="${APP_ROOT}/venv/bin:/usr/local/bin:/usr/bin:/bin" \
  VIDEO_FACTORY_SAU_HOME="${STATE_ROOT}/runtime/social-auto-upload" \
  VIDEO_FACTORY_INSTALL_MANAGED_CHROMIUM=1 \
  VIDEO_FACTORY_CHROME_PATH= \
  CHROME_PATH= \
  "${APP_ROOT}/venv/bin/video-factory" \
  --workspace "${STATE_ROOT}/workspace" publisher setup

browser_path=""
while IFS= read -r candidate; do
  browser_path="${candidate}"
  break
done < <(find "${STATE_ROOT}/runtime/social-auto-upload/browsers" -type f \
  \( -name chrome -o -name chromium \) -perm /111 2>/dev/null | sort)
if [[ -n "${browser_path}" ]]; then
  ln -sfn "${browser_path}" "${APP_ROOT}/bin/chromium"
else
  echo "warning: managed Chromium executable was not found" >&2
fi

if [[ "${VIDEO_FACTORY_INSTALL_YOUTUBE_RUNTIME:-0}" == "1" ]]; then
  runuser -u "${APP_USER}" -- env \
    HOME="${STATE_ROOT}" \
    PATH="${APP_ROOT}/venv/bin:/usr/local/bin:/usr/bin:/bin" \
    VIDEO_FACTORY_YOUTUBE_RUNTIME_HOME="${STATE_ROOT}/runtime/youtube" \
    "${APP_ROOT}/venv/bin/video-factory" youtube-runtime setup
fi

install -o root -g root -m 0644 "${APP_ROOT}/app/deploy/oci/systemd/"*.service /etc/systemd/system/
install -o root -g root -m 0644 "${APP_ROOT}/app/deploy/oci/systemd/"*.timer /etc/systemd/system/
systemctl daemon-reload

chown -R "${APP_USER}:${APP_USER}" "${STATE_ROOT}" "${APP_ROOT}/bin"
chmod 0750 "${APP_ROOT}/app/deploy/oci/"*.sh

echo
echo "Oracle deployment installed."
echo "1. Edit ${env_target} and add provider credentials."
echo "2. Run: sudo -u ${APP_USER} ${APP_ROOT}/app/deploy/oci/smoke-test.sh"
echo "3. Log in to Video Accounts using login-desktop.sh."
echo "4. Enable timers only after the smoke test succeeds."
