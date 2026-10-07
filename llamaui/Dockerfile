# llama.cpp's web UI in front of the vLLM server (see proxy.py).
# The UI is llama.cpp's own prebuilt release asset, pinned and checksum-verified;
# bump both args together (sha256 from <version>/dist.tar.gz.sha256 in the bucket).
FROM python:3.12-slim

ARG LLAMA_UI_VERSION=b11457
ARG LLAMA_UI_SHA256=f223cd450351edf41f941e632c4908a8d4cd024df18430e8b8aa56f78d4cda62

ADD --checksum=sha256:${LLAMA_UI_SHA256} \
    https://huggingface.co/buckets/ggml-org/llama-ui/resolve/${LLAMA_UI_VERSION}/dist.tar.gz?download=true \
    /tmp/dist.tar.gz
RUN mkdir -p /ui && tar -xzf /tmp/dist.tar.gz -C /ui && rm /tmp/dist.tar.gz \
    && if [ ! -f /ui/index.html ]; then d=$(dirname "$(find /ui -name index.html | head -1)"); mv "$d"/* /ui/; fi \
    && test -f /ui/index.html

COPY proxy.py /app/proxy.py
EXPOSE 8080
CMD ["python", "-u", "/app/proxy.py"]
