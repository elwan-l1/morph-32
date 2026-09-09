"""Loopback OpenAI-compatible serving through the pinned MLX-LM HTTP stack."""

import argparse
import json
import logging
from dataclasses import replace
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.server import APIHandler, ModelProvider, ResponseGenerator

from .runtime import load

DEFAULT_MODEL_ID = "qwen3.8-27b-morph32-3s"


class MorphModelProvider(ModelProvider):
    """Expose one verified MORPH checkpoint, without remote or native fallback loading."""

    def __init__(self, args, checkpoint, model_id):
        super().__init__(args)
        self.checkpoint = Path(checkpoint).resolve()
        self.model_id = model_id
        self._model_map[model_id] = str(self.checkpoint)

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if Path(model_path).resolve() != self.checkpoint:
            raise ValueError(f"This server only provides {self.model_id}")
        if adapter_path is not None or draft_model_path is not None:
            raise ValueError("Adapters and draft models are not supported")
        model, tokenizer, manifest = load(self.checkpoint, verify=True)
        self.model = model
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.draft_model = None
        self.model_key = (model_path, None, None)
        # Keep a single generation stream and the tested single-request cache path.
        self.is_batchable = False
        logging.info(
            "Loaded %s; integrity verified; tools=%s", self.model_id, tokenizer.has_tool_calling
        )


class MorphResponseGenerator(ResponseGenerator):
    def _tokenize(self, tokenizer, request, args):
        args = replace(
            args,
            chat_template_kwargs={**(args.chat_template_kwargs or {}), "enable_thinking": False},
        )
        result = super()._tokenize(tokenizer, request, args)
        if len(result[0]) + args.max_tokens > self.cli_args.context_length:
            raise ValueError(
                f"Prompt plus output budget exceeds {self.cli_args.context_length} tokens"
            )
        return result


class MorphAPIHandler(APIHandler):
    def do_POST(self):
        try:
            super().do_POST()
        except (ValueError, TypeError, KeyError, AssertionError) as error:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(error)}}).encode())

    def validate_model_parameters(self):
        super().validate_model_parameters()
        self._validate(
            "max_tokens", int, min_val=1, max_val=self.response_generator.cli_args.max_tokens
        )

    def handle_models_request(self):
        model_id = self.response_generator.model_provider.model_id
        response = {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": self.created, "owned_by": "morph32"}
            ],
        }
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def handle_health_check(self):
        loaded = self.response_generator.model_provider.model is not None
        self._set_completion_headers(200 if loaded else 503)
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok" if loaded else "loading"}).encode())


def serve(
    checkpoint, *, port=8081, model_id=DEFAULT_MODEL_ID, max_tokens=4096, context_length=32768
):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not 1 <= port <= 65535 or not 1 <= max_tokens < context_length:
        raise ValueError("Require valid port and 0 < max_tokens < context_length")
    args = argparse.Namespace(
        model=str(checkpoint),
        adapter_path=None,
        draft_model=None,
        pipeline=False,
        trust_remote_code=False,
        chat_template=None,
        use_default_chat_template=False,
        chat_template_args={"enable_thinking": False},
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=max_tokens,
        num_draft_tokens=0,
        decode_concurrency=1,
        prompt_concurrency=1,
        prefill_step_size=512,
        prompt_cache_size=2,
        prompt_cache_bytes=512 * 1024**2,
        allowed_origins=["http://127.0.0.1", "http://localhost"],
        context_length=context_length,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = MorphModelProvider(args, checkpoint, model_id)
    # Fail immediately on a corrupt checkpoint, before advertising a listening endpoint.
    provider.load_default()
    generator = MorphResponseGenerator(provider, LRUPromptCache(args.prompt_cache_size))
    try:
        with ThreadingHTTPServer(
            ("127.0.0.1", port), partial(MorphAPIHandler, generator)
        ) as server:
            logging.info("Serving %s at http://127.0.0.1:%s/v1 (non-thinking)", model_id, port)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        generator.stop_and_join()
