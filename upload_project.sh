#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-10.15.171.204}"
REMOTE_PORT="${REMOTE_PORT:-30428}"
REMOTE_DIR="${REMOTE_DIR:-/2024571007/iml/qwen_finetune}"
REMOTE_PASSWORD='x^Uj5qNZi3sOi6UWv*p9iudiH$mHq6po'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/README.md" && -d "${SCRIPT_DIR}/src" ]]; then
  PROJECT_DIR="${SCRIPT_DIR}"
else
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if ! command -v scp >/dev/null 2>&1; then
  echo "Error: scp is required but was not found." >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "Error: sshpass is required but was not found." >&2
  echo "Install it first, for example: sudo apt install sshpass" >&2
  exit 1
fi

if [[ "${REMOTE_PASSWORD}" == "请把这里改成你的SSH密码" || -z "${REMOTE_PASSWORD}" ]]; then
  echo "Error: please set REMOTE_PASSWORD in this script first." >&2
  exit 1
fi

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
CONTROL_PATH="${TMPDIR:-/tmp}/qwen_finetune_ssh_${REMOTE_HOST}_${REMOTE_PORT}_${REMOTE_USER}_$$"
SSH_OPTS=(
  -p "${REMOTE_PORT}"
  -o ControlMaster=auto
  -o ControlPath="${CONTROL_PATH}"
  -o ControlPersist=10m
)
SCP_OPTS=(
  -P "${REMOTE_PORT}"
  -o ControlMaster=auto
  -o ControlPath="${CONTROL_PATH}"
  -o ControlPersist=10m
)
UPLOAD_ITEMS=(
  "data"
  "README.md"
  "src"
  "scripts"
)

echo "Uploading project:"
echo "  local : ${PROJECT_DIR}/"
echo "  remote: ${SSH_TARGET}:${REMOTE_DIR}/"

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "${SSH_TARGET}" >/dev/null 2>&1 || true
  rm -f "${CONTROL_PATH}"
}
trap cleanup EXIT

echo "Opening SSH connection"
sshpass -p "${REMOTE_PASSWORD}" ssh "${SSH_OPTS[@]}" -Nf "${SSH_TARGET}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "mkdir -p '${REMOTE_DIR}'"

UPLOAD_PATHS=()
for item in "${UPLOAD_ITEMS[@]}"; do
  local_path="${PROJECT_DIR}/${item}"
  if [[ -e "${local_path}" ]]; then
    UPLOAD_PATHS+=("${local_path}")
  else
    echo "Skipping missing item: ${item}"
  fi
done

if [[ "${#UPLOAD_PATHS[@]}" -eq 0 ]]; then
  echo "Error: no upload items were found." >&2
  exit 1
fi

echo "Uploading selected files and directories"
sshpass -p "${REMOTE_PASSWORD}" scp -r "${SCP_OPTS[@]}" "${UPLOAD_PATHS[@]}" "${SSH_TARGET}:${REMOTE_DIR}/"

echo "Remote directory contents:"
sshpass -p "${REMOTE_PASSWORD}" ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "ls '${REMOTE_DIR}'"

echo "Upload complete."
