"""Serve llama.cpp's web UI in front of a vLLM OpenAI-compatible server.

The llama.cpp UI expects a few llama-server-only things that vLLM does not have:
  - GET /props            -> synthesized here (context size, chat template, defaults)
  - "timings" in each SSE  -> computed here from wall clock + vLLM's running usage,
                              which is what makes the UI show tokens/s
  - delta.reasoning_content -> vLLM may send it as delta.reasoning; renamed here
Everything llama-server-specific that has no vLLM equivalent (slots, tools, MCP,
resumable streams, model load/unload) answers like a llama-server without that feature.

stdlib only. Env: UPSTREAM (default http://vllm:18020, the compose alias of the
single/batch service), UI_DIR, MODELS_DIR (vLLM's /app/models mounted here, for the
chat template), PORT.
"""
import http.client
import json
import mimetypes
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = urllib.parse.urlparse(os.environ.get("UPSTREAM", "http://vllm:18020"))
UI_DIR = os.path.realpath(os.environ.get("UI_DIR", "/ui"))
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
PORT = int(os.environ.get("PORT", "8080"))

# llama.cpp sampler knobs vLLM does not know; dropped so its log stays quiet
LLAMA_ONLY = {
    "samplers", "backend_sampling", "timings_per_token", "return_progress", "sse_ping_interval",
    "reasoning_format", "dynatemp_range", "dynatemp_exponent", "xtc_probability", "xtc_threshold",
    "typ_p", "repeat_last_n", "dry_multiplier", "dry_base", "dry_allowed_length",
    "dry_penalty_last_n", "cache_prompt", "n_probs", "post_sampling_probs", "id_slot",
}


def upstream(method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(UPSTREAM.hostname, UPSTREAM.port or 80, timeout=3600)
    conn.request(method, path, body=body, headers=headers or {})
    return conn, conn.getresponse()


def model_info():
    conn, r = upstream("GET", "/v1/models")
    data = json.loads(r.read())["data"][0]
    conn.close()
    return data


def read_model_file(root, name):
    rel = os.path.relpath(root, "/app/models") if root.startswith("/app/models") else os.path.basename(root)
    try:
        with open(os.path.join(MODELS_DIR, rel, name), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def build_props():
    m = model_info()
    root = m.get("root", "")
    gen = {}
    try:
        gen = json.loads(read_model_file(root, "generation_config.json") or "{}")
    except ValueError:
        pass
    params = {
        "n_predict": -1, "seed": -1, "temperature": gen.get("temperature", 0.7),
        "top_k": gen.get("top_k", 20), "top_p": gen.get("top_p", 0.95), "min_p": 0.0,
        "dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_n_sigma": -1.0,
        "xtc_probability": 0.0, "xtc_threshold": 0.1, "typ_p": 1.0, "repeat_last_n": 64,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": -1,
        "dry_sequence_breakers": [], "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
        "stop": [], "max_tokens": -1, "n_keep": 0, "n_discard": 0, "ignore_eos": False,
        "stream": True, "logit_bias": [], "n_probs": 0, "min_keep": 0, "grammar": "",
        "grammar_lazy": False, "grammar_triggers": [], "preserved_tokens": [],
        "chat_format": "vLLM", "reasoning_format": "deepseek", "reasoning_in_content": False,
        "generation_prompt": "", "samplers": ["top_k", "top_p", "temperature"],
        "backend_sampling": False, "speculative.n_max": 7, "speculative.n_min": 0,
        "speculative.p_min": 0.0, "timings_per_token": False, "post_sampling_probs": False,
        "lora": [],
    }
    return {
        "default_generation_settings": {
            "id": 0, "id_task": -1, "n_ctx": m.get("max_model_len", 0), "speculative": True,
            "is_processing": False, "params": params, "prompt": "",
            "next_token": {"has_next_token": True, "has_new_line": False, "n_remain": -1,
                           "n_decoded": 0, "stopping_word": ""},
        },
        "total_slots": 8,
        "model_path": m["id"],
        "model_alias": m["id"],
        "role": "model",
        "modalities": {"vision": False, "audio": False, "video": False},
        "chat_template": read_model_file(root, "chat_template.jinja"),
        "bos_token": "", "eos_token": "",
        "build_info": "vLLM via vllm-llamaui proxy",
    }


class Timer:
    """Turns vLLM's running usage into llama-server style timings."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.t_first = None
        self.prompt_n = self.predicted_n = self.cache_n = 0

    def see(self, ev):
        u = ev.get("usage")
        if u:
            self.prompt_n = u.get("prompt_tokens", self.prompt_n)
            self.predicted_n = u.get("completion_tokens", self.predicted_n)
            self.cache_n = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if self.t_first is None and self.predicted_n > 0:
            self.t_first = time.perf_counter()

    def timings(self):
        now = time.perf_counter()
        first = self.t_first or now
        prompt_ms = (first - self.t0) * 1000
        predicted_ms = (now - first) * 1000
        # the first chunk's tokens arrive with the prefill; time the rest
        n = max(self.predicted_n - 1, 0)
        return {
            "cache_n": self.cache_n, "prompt_n": self.prompt_n - self.cache_n,
            "prompt_ms": prompt_ms,
            "prompt_per_second": (self.prompt_n - self.cache_n) / prompt_ms * 1000 if prompt_ms else 0,
            "predicted_n": self.predicted_n, "predicted_ms": predicted_ms,
            "predicted_per_token_ms": predicted_ms / n if n else 0,
            "predicted_per_second": n / predicted_ms * 1000 if predicted_ms else 0,
        }


def fix_delta(ev):
    for c in ev.get("choices") or []:
        for d in (c.get("delta"), c.get("message")):
            if d and d.get("reasoning") and not d.get("reasoning_content"):
                d["reasoning_content"] = d.pop("reasoning")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # metadata only, never bodies
        print(f"{self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}", flush=True)

    def send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def fwd_headers(self):
        h = {"Content-Type": "application/json"}
        if self.headers.get("Authorization"):
            h["Authorization"] = self.headers["Authorization"]
        return h

    def passthrough(self, method, body=None):
        conn, r = upstream(method, self.path, body, self.fwd_headers())
        data = r.read()
        conn.close()
        self.send_response(r.status)
        self.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def unavailable(self, e):
        # vLLM still loading (first start takes minutes) or stopped
        self.send_json(503, {"error": {"code": 503, "type": "unavailable_error",
                                       "message": f"vLLM at {UPSTREAM.geturl()} not reachable yet: {e}"}})

    # ---- routes -------------------------------------------------------------
    def do_GET(self):
        try:
            self.route_get()
        except (ConnectionError, OSError, http.client.HTTPException) as e:
            self.unavailable(e)

    def do_POST(self):
        try:
            self.route_post()
        except (ConnectionError, OSError, http.client.HTTPException) as e:
            self.unavailable(e)

    def route_get(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/props":
            return self.send_json(200, build_props())
        if path in ("/health", "/v1/models", "/metrics"):
            return self.passthrough("GET")
        if path == "/models":
            self.path = "/v1/models"
            return self.passthrough("GET")
        if path in ("/slots", "/tools") or path.startswith(("/mcp", "/cors-proxy", "/v1/stream", "/models/")):
            return self.send_json(501, {"error": {"code": 501, "message": "not supported by vLLM", "type": "not_supported_error"}})
        self.static(path)

    def route_post(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/streams/lookup":
            self.body()
            return self.send_json(200, [])
        if path == "/v1/chat/completions":
            return self.chat()
        if path in ("/v1/completions", "/tokenize", "/detokenize"):
            return self.passthrough("POST", self.body())
        self.body()
        self.send_json(501, {"error": {"code": 501, "message": "not supported by vLLM", "type": "not_supported_error"}})

    def static(self, path):
        rel = urllib.parse.unquote(path).lstrip("/") or "index.html"
        full = os.path.realpath(os.path.join(UI_DIR, rel))
        if not full.startswith(UI_DIR + os.sep) or not os.path.isfile(full):
            full = os.path.join(UI_DIR, "index.html")  # SPA fallback
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(full)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def chat(self):
        req = json.loads(self.body() or b"{}")
        for k in LLAMA_ONLY:
            req.pop(k, None)
        stream = bool(req.get("stream"))
        if stream:
            req["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        timer = Timer()
        conn, r = upstream("POST", "/v1/chat/completions", json.dumps(req).encode(), self.fwd_headers())

        if r.status != 200 or not stream:
            data = r.read()
            conn.close()
            if r.status == 200:
                ev = json.loads(data)
                timer.see(ev)
                timer.t_first = timer.t_first or timer.t0
                fix_delta(ev)
                ev["timings"] = timer.timings()
                data = json.dumps(ev).encode()
            self.send_response(r.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(line):
            b = line + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()

        try:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    emit(b"data: [DONE]")
                    continue
                ev = json.loads(payload)
                timer.see(ev)
                fix_delta(ev)
                ev["timings"] = timer.timings()
                emit(b"data: " + json.dumps(ev).encode())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hit stop; closing upstream aborts the generation
        finally:
            conn.close()


if __name__ == "__main__":
    print(f"llama.cpp UI on :{PORT} -> vLLM {UPSTREAM.geturl()}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
