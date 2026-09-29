# Training / inference image. CPU by default; swap the base image for a CUDA one to train PatchTST on GPU.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[torch,bayes]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# data and artifacts are mounted at runtime:
#   docker run -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts m5 build-db
ENTRYPOINT ["m5"]
CMD ["--help"]
