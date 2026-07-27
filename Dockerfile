FROM vllm/vllm-openai:v0.10.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsm6 \
    libxext6 \
    && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt requirements-core.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . /app
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
