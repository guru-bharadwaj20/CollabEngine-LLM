# Reproduce the analysis exactly; reproduce the generation approximately.
#
# The distinction is not hedging, it is the truth about what a container can do
# here. Everything downstream of a transcript -- scoring, integrity, the gate,
# the interaction test, the equivalence bounds, the figures -- is deterministic
# and is reproduced bit-for-bit by this image. Generation is not: the corpus was
# produced on Windows against a llama.cpp CUDA build, and llama.cpp is
# deterministic per slot but not across continuous-batching arrangements
# (docs/LLAMACPP-SETUP.md). A container that claimed otherwise would be making
# the same kind of claim this paper exists to criticise.
#
#   docker build -t collabengine .
#   docker run --rm -v "$PWD/runs:/work/runs" collabengine \
#       python scripts/analysis/gate_report.py
#
# To generate rather than analyse, the image needs a GPU and the weights:
#   docker run --gpus all -v "$PWD/models:/work/models" -v "$PWD/runs:/work/runs" \
#       collabengine bash scripts/experiments/rebuild-corpus.sh
#
# Weights are not baked in: 44 GiB of GGUF has no business in a layer, and
# `scripts/ops/fetch_models.py` pins the repo and filename for every one of
# them. `scripts/ops/env_stamp.py` records the sha256 of the file actually
# served, which is the only thing that proves which conversion was used.

FROM python:3.10.20-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git is here for `env_stamp.py`, which records the commit a run came from. An
# image without it produces stamps with a null commit, which is the failure this
# project has already had once with a null slot size.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# The lockfile first, so a source edit does not invalidate the dependency layer.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docs/ docs/

# The suite is the acceptance test for the image: 379 tests, seconds, no GPU and
# no corpus. If it does not pass here, nothing this image reports can be
# trusted, and finding that out at build time costs nothing.
RUN python -m pytest -q

CMD ["python", "scripts/analysis/gate_report.py"]
