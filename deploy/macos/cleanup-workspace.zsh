#!/bin/zsh
set -eu

PROJECT_ROOT="/Users/clairehou/pyProjects/video_factory"
WORKSPACE="${PROJECT_ROOT}/workspace"
TRASH_ROOT="/Users/clairehou/.Trash/video-factory-cleanup-20260829-1832"

if [[ "$(pwd -P)" != "${PROJECT_ROOT}" || ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  echo "refusing cleanup outside the expected Video Factory project" >&2
  exit 2
fi
if [[ -e "${TRASH_ROOT}" ]]; then
  echo "trash destination already exists: ${TRASH_ROOT}" >&2
  exit 2
fi

mkdir -p "${TRASH_ROOT}"

move_to_trash() {
  local source="$1"
  if [[ ! -e "${source}" ]]; then
    return
  fi
  local relative="${source#${PROJECT_ROOT}/}"
  local destination="${TRASH_ROOT}/${relative}"
  mkdir -p "${destination:h}"
  mv "${source}" "${destination}"
  echo "moved ${relative}"
}

# Local imports were already content-addressed under workspace/assets; the
# collection manifests reference those archived copies, not these duplicates.
for source in \
  "${WORKSPACE}/imports/zCJtYuqwm7E.mp4" \
  "${WORKSPACE}/imports/zCJtYuqwm7E.1080p.mkv" \
  "${WORKSPACE}/imports/zCJtYuqwm7E.en-orig.json3" \
  "${WORKSPACE}/imports/incomplete/zCJtYuqwm7E.1080p.f299.mp4.part" \
  "${WORKSPACE}/renders/youtube-zCJtYuqwm7E-34467e80" \
  "${WORKSPACE}/renders/youtube-zCJtYuqwm7E-6378029d"
do
  move_to_trash "${source}"
done

# These are deterministic render intermediates. Final MP4s and all three SRT
# variants stay in place, as do current and superseded collection manifests.
while IFS= read -r -d '' directory; do
  move_to_trash "${directory}"
done < <(find "${WORKSPACE}/renders" -depth -type d \
  \( -name '*-frames' -o -name '*.frames' \) -print0)

while IFS= read -r -d '' file; do
  move_to_trash "${file}"
done < <(find "${WORKSPACE}/renders" -type f \
  \( -name '*.ffconcat' -o -name '*.filters.txt' -o -name '.DS_Store' \) -print0)

echo "recoverable cleanup stored at ${TRASH_ROOT}"
