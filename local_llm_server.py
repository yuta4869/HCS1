#!/usr/bin/env python3
"""
Simple local LLM server that exposes a minimal OpenAI-compatible API.
This avoids the Heavy FastAPI/uvicorn runtime so it can run in restricted environments.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

try:
    import ctypes

    # Ensure the system libstdc++ is loaded so llama_cpp can find GLIBCXX_3.4.30 even
    # if the Python interpreter comes from an older Conda environment.
    ctypes.CDLL(
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        mode=ctypes.RTLD_GLOBAL,  # type: ignore[attr-defined]
    )
except Exception as e:  # pragma: no cover - best effort safeguard
    print(f"[SimpleLLM] Warning: failed to preload system libstdc++: {e}", file=sys.stderr)

from llama_cpp import Llama  # noqa: E402


class LLMHTTPHandler(BaseHTTPRequestHandler):
    server_version = "SimpleLLM/0.1"
    protocol_version = "HTTP/1.1"

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - reduce noise
        sys.stdout.write(
            "[SimpleLLM] %s - - %s\n"
            % (self.log_date_time_string(), fmt % args)
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            response = {
                "object": "list",
                "data": [
                    {
                        "id": self.server.model_id,
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            }
            self._json_response(200, response)
            return
        self._json_response(404, {"error": {"message": f"Unknown path: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json_response(404, {"error": {"message": f"Unknown path: {self.path}"}})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._json_response(400, {"error": {"message": "Invalid JSON body"}})
            return

        messages = payload.get("messages")
        if not isinstance(messages, list):
            self._json_response(400, {"error": {"message": "'messages' must be a list"}})
            return

        max_tokens = payload.get("max_tokens", 256)
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.95)
        model_name = payload.get("model", self.server.model_id)

        try:
            completion = self.server.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            content = completion["choices"][0]["message"]["content"]
            response_body = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": completion["choices"][0].get("finish_reason", "stop"),
                    }
                ],
                "usage": completion.get("usage", {}),
            }
            self._json_response(200, response_body)
        except Exception as e:
            self._json_response(
                500,
                {"error": {"message": f"LLM inference failed: {e}"}},
            )


class SimpleLLMServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, llm: Llama, model_id: str):
        super().__init__(server_address, RequestHandlerClass)
        self.llm = llm
        self.model_id = model_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple local LLM server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    print("[SimpleLLM] Loading model...")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.threads,
    )
    model_id = os.path.basename(args.model)
    print(f"[SimpleLLM] Model loaded: {model_id}")
    print(f"[SimpleLLM] Serving on http://{args.host}:{args.port}")

    server = SimpleLLMServer((args.host, args.port), LLMHTTPHandler, llm, model_id)

    def shutdown(signum=None, frame=None):
        print(f"[SimpleLLM] Received signal {signum}, shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[SimpleLLM] Server stopped.")


if __name__ == "__main__":
    main()
