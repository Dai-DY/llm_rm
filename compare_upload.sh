#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-10.15.171.204}"
REMOTE_PORT="${REMOTE_PORT:-30428}"
REMOTE_DIR="${REMOTE_DIR:-/2024571007/iml/qwen_finetune}"
REMOTE_PASSWORD='x^Uj5qNZi3sOi6UWv*p9iudiH$mHq6po'

UPLOAD_ITEMS=(
  "README.md"
  "src"
  "scripts"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/README.md" && -d "${SCRIPT_DIR}/src" ]]; then
  PROJECT_DIR="${SCRIPT_DIR}"
else
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "Error: sshpass is required but was not found." >&2
  echo "Install it first, for example: sudo apt install sshpass" >&2
  exit 1
fi

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "Error: sha256sum is required but was not found." >&2
  exit 1
fi

if [[ "${REMOTE_PASSWORD}" == "请把这里改成你的SSH密码" || -z "${REMOTE_PASSWORD}" ]]; then
  echo "Error: please set REMOTE_PASSWORD in this script first." >&2
  exit 1
fi

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTS=(-p "${REMOTE_PORT}")
TMP_DIR="$(mktemp -d)"
LOCAL_MANIFEST="${TMP_DIR}/local.tsv"
REMOTE_MANIFEST="${TMP_DIR}/remote.tsv"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

make_local_manifest() {
  : > "${LOCAL_MANIFEST}"

  for item in "${UPLOAD_ITEMS[@]}"; do
    local_path="${PROJECT_DIR}/${item}"
    if [[ ! -e "${local_path}" ]]; then
      echo "Local missing item: ${item}" >&2
      continue
    fi

    if [[ -d "${local_path}" ]]; then
      (
        cd "${PROJECT_DIR}"
        find "${item}" -type f -print0
      )
    else
      printf '%s\0' "${item}"
    fi
  done |
    while IFS= read -r -d '' rel_path; do
      size="$(stat -c '%s' "${PROJECT_DIR}/${rel_path}")"
      hash="$(sha256sum "${PROJECT_DIR}/${rel_path}" | awk '{print $1}')"
      printf '%s\t%s\t%s\n' "${rel_path}" "${size}" "${hash}"
    done |
    sort > "${LOCAL_MANIFEST}"
}

make_remote_manifest() {
  sshpass -p "${REMOTE_PASSWORD}" ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" \
    "bash -s -- $(printf '%q' "${REMOTE_DIR}") $(printf '%q ' "${UPLOAD_ITEMS[@]}")" \
    <<'REMOTE_SCRIPT' |
set -euo pipefail

REMOTE_DIR="$1"
shift
UPLOAD_ITEMS=("$@")

if [[ ! -d "${REMOTE_DIR}" ]]; then
  exit 0
fi

cd "${REMOTE_DIR}"

for item in "${UPLOAD_ITEMS[@]}"; do
  if [[ ! -e "${item}" ]]; then
    echo "Remote missing item: ${item}" >&2
    continue
  fi

  if [[ -d "${item}" ]]; then
    find "${item}" -type f -print0
  else
    printf '%s\0' "${item}"
  fi
done |
  while IFS= read -r -d '' rel_path; do
    size="$(stat -c '%s' "${rel_path}")"
    hash="$(sha256sum "${rel_path}" | awk '{print $1}')"
    printf '%s\t%s\t%s\n' "${rel_path}" "${size}" "${hash}"
  done |
  sort
REMOTE_SCRIPT
    sort > "${REMOTE_MANIFEST}"
}

echo "Comparing upload files:"
echo "  local : ${PROJECT_DIR}/"
echo "  remote: ${SSH_TARGET}:${REMOTE_DIR}/"

make_local_manifest
make_remote_manifest

only_local="$(comm -23 <(cut -f1 "${LOCAL_MANIFEST}") <(cut -f1 "${REMOTE_MANIFEST}") || true)"
only_remote="$(comm -13 <(cut -f1 "${LOCAL_MANIFEST}") <(cut -f1 "${REMOTE_MANIFEST}") || true)"
changed="$(
  join -t $'\t' -j 1 "${LOCAL_MANIFEST}" "${REMOTE_MANIFEST}" |
    awk -F '\t' '$2 != $4 || $3 != $5 {print $1}'
)"
same_count="$(
  join -t $'\t' -j 1 "${LOCAL_MANIFEST}" "${REMOTE_MANIFEST}" |
    awk -F '\t' '$2 == $4 && $3 == $5 {count++} END {print count + 0}'
)"

print_section() {
  local title="$1"
  local content="$2"

  echo
  echo "${title}"
  if [[ -n "${content}" ]]; then
    printf '%s\n' "${content}"
  else
    echo "  none"
  fi
}

print_section "Only in local, will be uploaded as new files:" "${only_local}"
print_section "Only in remote, upload script will not delete these:" "${only_remote}"
print_section "Different content, upload will overwrite remote files:" "${changed}"

echo
echo "Same files: ${same_count}"
echo "Compare complete."
