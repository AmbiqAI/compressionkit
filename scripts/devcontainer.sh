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
#   scripts/devcontainer.sh gpu-check            # verify the GPU is actually usable (not just nvidia-smi)
#   scripts/devcontainer.sh gpu-recover          # auto-heal a broken GPU: restart, else full recreate
#
# Examples:
#   scripts/devcontainer.sh exec -- uv run pytest tests/test_two_stage.py -q
#   scripts/devcontainer.sh exec -- uv run python scripts/train_rvq_prior.py --help
#
# Why gpu-check/gpu-recover instead of just running `nvidia-smi`:
#   `nvidia-smi` inside the container only proves the NVML library can talk to
#   the driver — it does NOT prove a CUDA context can actually be created on
#   the container's own device file descriptors. After a host-side kernel/
#   driver churn event (e.g. an unattended-upgrades kernel update rebuilding
#   the NVIDIA DKMS module without a reboot), `nvidia-smi` can keep working
#   while real framework calls (TensorFlow/PyTorch CUDA init) fail inside an
#   already-running container. gpu-check does a real minimal CUDA init via
#   TensorFlow so it catches that failure mode; gpu-recover tries the cheap
#   fix first (`docker restart`, which re-runs the NVIDIA container-runtime
#   hook and re-binds fresh device state) before falling back to removing
#   and recreating the container entirely via `devcontainer up`.

set -euo pipefail

# Resolve the main worktree path: the dev container is created against the
# primary checkout, not any linked `git worktree add` copy. `git worktree
# list` always prints the main worktree first.
resolve_main_worktree() {
    if git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # Strip only the `worktree ` prefix and print the rest of the line
        # verbatim (not `{print $2}`, which splits on whitespace and would
        # silently truncate a workspace path containing spaces).
        git -C "$(dirname "${BASH_SOURCE[0]}")" worktree list --porcelain \
            | awk '/^worktree /{sub(/^worktree /, ""); print; exit}'
    fi
}

MAIN_WORKSPACE="${DEVCONTAINER_WORKSPACE:-$(resolve_main_worktree)}"
if [[ -z "${MAIN_WORKSPACE}" ]]; then
    echo "error: could not resolve main worktree path; set DEVCONTAINER_WORKSPACE" >&2
    exit 1
fi

container_id() {
    local ids
    ids="$(docker ps -q --filter "label=devcontainer.local_folder=${MAIN_WORKSPACE}")"
    if [[ -n "${ids}" ]] && [[ "$(wc -l <<< "${ids}")" -gt 1 ]]; then
        echo "error: multiple running containers match workspace ${MAIN_WORKSPACE}:" >&2
        echo "${ids}" >&2
        echo "Disambiguate manually with --container-id, or stop the extras." >&2
        exit 1
    fi
    echo "${ids}"
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

cmd_gpu_check() {
    local id
    id="$(cmd_id)"
    echo "-- nvidia-smi (NVML-level check) --"
    if ! devcontainer exec --container-id "${id}" --workspace-folder "${MAIN_WORKSPACE}" -- nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; then
        echo "gpu-check: FAILED at nvidia-smi (driver/NVML not visible in container)" >&2
        return 1
    fi
    echo "-- CUDA context init via TensorFlow (real usability check) --"
    if ! devcontainer exec --container-id "${id}" --workspace-folder "${MAIN_WORKSPACE}" -- \
        uv run python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'no GPU visible to TensorFlow'
with tf.device('/GPU:0'):
    x = tf.constant([1.0, 2.0]) + tf.constant([3.0, 4.0])
    x.numpy()
print('CUDA context OK:', gpus)
" 2>&1 | tail -20; then
        echo "gpu-check: FAILED — nvidia-smi worked but a real CUDA context could not be created." >&2
        echo "This is the 'stale device binding' failure mode; run '$(basename "$0") gpu-recover'." >&2
        return 1
    fi
    echo "gpu-check: OK"
}

cmd_gpu_recover() {
    local id
    id="$(container_id)"
    if [[ -z "${id}" ]]; then
        echo "no running container; bringing one up..."
        cmd_up
        # Bringing the container up only proves it was created — it says
        # nothing about whether the GPU/CUDA issue gpu-recover exists to fix
        # is actually resolved. Verify with gpu-check, same as the other two
        # recovery paths below, so a caller checking this function's exit
        # code gets a real answer instead of a false "success".
        cmd_gpu_check
        return
    fi
    echo "Attempting cheap recovery: docker restart (re-runs the NVIDIA container-runtime hook)..."
    docker restart "${id}" >/dev/null
    sleep 3
    if cmd_gpu_check; then
        echo "gpu-recover: fixed via docker restart."
        return 0
    fi
    echo "docker restart did not fix it; falling back to a full recreate (stop + remove + devcontainer up)..."
    # `docker stop` alone is not enough: `devcontainer up` finds and restarts
    # an existing (even stopped) container for this workspace rather than
    # creating a fresh one, so the stale device bindings this fallback exists
    # to clear would otherwise survive. Remove the container so `up` has no
    # choice but to create a new one from the image.
    docker stop "${id}" >/dev/null 2>&1 || true
    docker rm "${id}" >/dev/null 2>&1 || true
    cmd_up
    cmd_gpu_check
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
    gpu-check) cmd_gpu_check ;;
    gpu-recover) cmd_gpu_recover ;;
    *)
        echo "usage: $(basename "$0") {up|id|exec -- <cmd>|shell|down|status|gpu-check|gpu-recover}" >&2
        exit 1
        ;;
esac
