ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY v2/vllm/ /usr/local/lib/python3.10/dist-packages/vllm/
COPY v3/vllm/ /usr/local/lib/python3.10/dist-packages/vllm/
ENV PDBLEND_ENGINE_TIMING=1 PDBLEND_ASYNC_IO=1
