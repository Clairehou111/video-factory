#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE="${VIDEO_FACTORY_ENV_FILE:-/etc/video-factory/video-factory.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

WORKSPACE="${VIDEO_FACTORY_WORKSPACE:-/srv/video-factory/workspace}"
BACKUP_DIR="${VIDEO_FACTORY_BACKUP_DIR:-/srv/video-factory/backups}"
RETENTION_DAYS="${VIDEO_FACTORY_BACKUP_RETENTION_DAYS:-14}"
INCLUDE_FINALS="${VIDEO_FACTORY_BACKUP_INCLUDE_FINALS:-0}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
temporary="$(mktemp -d)"
archive="${BACKUP_DIR}/video-factory-${timestamp}.tar.gz"

cleanup() {
  rm -rf "${temporary}"
}
trap cleanup EXIT INT TERM

mkdir -p "${BACKUP_DIR}" "${temporary}/workspace"
if [[ -f "${WORKSPACE}/index.sqlite3" ]]; then
  sqlite3 "${WORKSPACE}/index.sqlite3" ".backup '${temporary}/workspace/index.sqlite3'"
fi

for directory in manifests publish discovery; do
  if [[ -d "${WORKSPACE}/${directory}" ]]; then
    cp -a "${WORKSPACE}/${directory}" "${temporary}/workspace/"
  fi
done

if [[ "${INCLUDE_FINALS}" == "1" && -d "${WORKSPACE}/jobs" ]]; then
  mkdir -p "${temporary}/workspace/jobs"
  while IFS= read -r -d '' video; do
    relative="${video#${WORKSPACE}/}"
    install -D -m 0640 "${video}" "${temporary}/workspace/${relative}"
  done < <(find "${WORKSPACE}/jobs" -type f \
    \( -name 'final*.mp4' -o -name '*final*.mp4' \) -print0)
fi

tar -C "${temporary}" -czf "${archive}" workspace
sha256sum "${archive}" >"${archive}.sha256"
find "${BACKUP_DIR}" -type f -name 'video-factory-*' -mtime "+${RETENTION_DAYS}" -delete

if [[ -n "${OCI_BACKUP_BUCKET:-}" ]]; then
  if ! command -v oci >/dev/null 2>&1; then
    echo "OCI_BACKUP_BUCKET is set but OCI CLI is not installed" >&2
    exit 1
  fi
  namespace_args=()
  if [[ -n "${OCI_BACKUP_NAMESPACE:-}" ]]; then
    namespace_args=(--namespace-name "${OCI_BACKUP_NAMESPACE}")
  fi
  oci os object put --force --bucket-name "${OCI_BACKUP_BUCKET}" \
    "${namespace_args[@]}" --name "$(basename "${archive}")" --file "${archive}"
  oci os object put --force --bucket-name "${OCI_BACKUP_BUCKET}" \
    "${namespace_args[@]}" --name "$(basename "${archive}.sha256")" --file "${archive}.sha256"
fi

echo "backup created: ${archive}"

