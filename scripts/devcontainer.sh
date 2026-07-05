#!/usr/bin/env bash
# Helper for driving the project's dev container from an agent/session that is
# running *outside* the container (e.g. a VS Code Remote-Containers window
# already has it open, or you're on a plain host shell / git worktree).
#
# Why this exists:
#   - `devcontainer` CLI looks up a running container by matching the
#     *workspace folder path* baked into its labels at creation time. If you
#     run it from a git worktree (a different path than the one the
#     container was originally created for), it fails with
#     "Dev container not found" even though the container is running.
#   - Plain `docker exec` works from anywhere, but silently drops the
#     `remoteEnv` values from devcontainer.json (e.g. TF_FORCE_GPU_ALLOW_GROWTH,
#     CUDA LD_LIBRARY_PATH additions) since it doesn't know about them.
#
# This script always resolves the *main* worktree checkout (the one the dev
# container was actually built against) and passes `--container-id` so
# commands work correctly regardless of which worktree/directory you're in.
#
# Usage:
#   scripts/devcontainer.sh up                  # bring the container up (idempotent)
#   scripts/devcontainer.sh id                   # print the running container id
#   scripts/devcontainer.sh exec -- <cmd> [args] # run a command in the container
#   scripts/devcontainer.sh shell                # open an interactive shell
#   scripts/devcontainer.sh down                 # stop the container
#   scripts/devcontainer.sh status               # show container state
#
# Examples:
#   scripts/devcontainer.sh exec -- uv run pytest tests/test_two_stage.py -q
#   scripts/devcontainer.sh exec -- uv run python scripts/train_rvq_prior.py --help

set -euo pipefail

# Resolve the main worktree path: the dev container is created against the
# primary checkout, not any linked `git worktree add` copy. `git worktree
# list` always prints the main worktree first.
resolve_main_worktree() {
    if git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$(dirname "${BASH_SOURCE[0]}")" worktree list --porcelain \
            | awk '/^worktree /{print $2; exit}'
    fi
}

MAIN_WORKSPACE="${DEVCONTAINER_WORKSPACE:-$(resolve_main_worktree)}"
if [[ -z "${MAIN_WORKSPACE}" ]]; then
    echo "error: could not resolve main worktree path; set DEVCONTAINER_WORKSPACE" >&2
    exit 1
fi

container_id() {
    docker ps -q --filter "label=devcontainer.local_folder=${MAIN_WORKSPACE}"
}

cmd_up() {
    devcontainer up --workspace-folder "${MAIN_WORKSPACE}"
}

cmd_id() {
    local id
    id="$(container_id)"
    if [[ -z "${id}" ]]; then
        echo "error: no running dev container for ${MAIN_WORKSPACE} (run 'up' first)" >&2
        exit 1
    fi
    echo "${id}"
}

cmd_exec() {
    local id
    id="$(cmd_id)"
    devcontainer exec --container-id "${id}" --workspace-folder "${MAIN_WORKSPACE}" -- "$@"
}

cmd_shell() {
    local id
    id="$(cmd_id)"
    devcontainer exec --container-id "${id}" --workspace-folder "${MAIN_WORKSPACE}" -- bash
}

cmd_down() {
    local id
    id="$(container_id)"
    if [[ -z "${id}" ]]; then
        echo "no running dev container for ${MAIN_WORKSPACE}"
        return 0
    fi
    docker stop "${id}"
}

cmd_status() {
    local id
    id="$(container_id)"
    if [[ -z "${id}" ]]; then
        echo "stopped: ${MAIN_WORKSPACE}"
        return 0
    fi
    docker ps --filter "id=${id}" --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}'
}

subcommand="${1:-}"
[[ $# -gt 0 ]] && shift || true

case "${subcommand}" in
    up) cmd_up ;;
    id) cmd_id ;;
    exec)
        [[ "${1:-}" == "--" ]] && shift
        cmd_exec "$@"
        ;;
    shell) cmd_shell ;;
    down) cmd_down ;;
    status) cmd_status ;;
    *)
        echo "usage: $(basename "$0") {up|id|exec -- <cmd>|shell|down|status}" >&2
        exit 1
        ;;
esac
