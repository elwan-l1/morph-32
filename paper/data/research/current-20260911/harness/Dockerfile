FROM ghcr.io/astral-sh/uv@sha256:4f5d923c9dcea037f57bda425dd209f3ec643da2f0b74227f68d09dab0b3bb36
RUN uv pip install --system numpy==2.2.6 tqdm==4.67.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/harness
WORKDIR /work
