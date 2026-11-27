#!/usr/bin/env bash
# Build the OpenFOAM image, or run a command in it from the repository root.
#   foam.sh build
#   foam.sh python examples/cfd/run.py list
# CFD_RUNS (default <repo>/runs) is mounted at /runs for case output.
set -euo pipefail

image="${CFD_IMAGE:-gp-omni-openfoam:2406}"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ "${1:-}" == "build" ]]; then
    exec docker build -t "$image" -f "$repository/examples/cfd/container/Dockerfile" "$repository"
fi

runs="${CFD_RUNS:-$repository/runs}"
mkdir -p "$runs"
name_option=()
[[ -n "${CFD_CONTAINER_NAME:-}" ]] && name_option=(--name "$CFD_CONTAINER_NAME")
exec docker run --rm "${name_option[@]}" \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --volume "$repository:/work" --volume "$runs:/runs" --workdir /work \
    "$image" bash -c 'cd /work && exec "$@"' foam "$@"  # the entry point changes directory
