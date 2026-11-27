# OpenCFD OpenFOAM v2406 with the repository's locked Python environment.
# Build with examples/cfd/container/foam.sh build. Torch is the locked version's CPU build,
# which avoids several gigabytes of CUDA libraries that the examples never use.
FROM opencfd/openfoam-default:2406

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libglu1-mesa libgl1 \
        libxrender1 libxcursor1 libxft2 libxinerama1 libxfixes3 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_CACHE_DIR=/tmp/uv_cache VIRTUAL_ENV=/opt/venv

WORKDIR /work
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY config ./config

RUN uv venv /opt/venv --python 3.12 \
    && uv export --locked --group dev --group openfoam --no-hashes --no-emit-project \
        > /tmp/locked.txt \
    && torch_version="$(sed -n 's/^torch==\([^ ;]*\).*/\1/p' /tmp/locked.txt | head -1)" \
    && grep -v -E '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' /tmp/locked.txt \
        > /tmp/without_torch.txt \
    && uv pip install "torch==${torch_version}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install -r /tmp/without_torch.txt \
    && uv pip install --no-deps -e . \
    && rm -rf /tmp/uv_cache /tmp/locked.txt /tmp/without_torch.txt \
    && chmod -R a+rX /opt/venv /opt/python

# The OpenFOAM entry point sources the solver environment before each command.
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1
