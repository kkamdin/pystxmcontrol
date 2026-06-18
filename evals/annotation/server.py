#!/usr/bin/env python3
"""Annotation server for STXM intelligence agent eval traces.

Usage:
    python server.py           # starts on http://localhost:7777
    python server.py 8080      # custom port
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).parent
EVALS_DIR = BASE_DIR.parent / "intelligence_agent"
LABELS_FILE = BASE_DIR / "labels.json"


def load_data():
    traces = {}
    with open(EVALS_DIR / "traces.jsonl") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                key = f"{t['run_id']}:{t['id']}"
                traces[key] = t

    results = {}
    with open(EVALS_DIR / "results.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = f"{r['run_id']}:{r['id']}"
                results[key] = r

    merged = []
    for key, trace in traces.items():
        result = results.get(key, {})
        merged.append({
            **trace,
            "assertions": result.get("assertions", {}),
            "response_chars": result.get("response_chars"),
        })

    merged.sort(key=lambda x: (x["run_id"], x["id"]))
    return merged


class Handler(BaseHTTPRequestHandler):
    _traces_cache = None

    def log_message(self, fmt, *args):
        pass  # suppress per-request logging

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/traces":
            if Handler._traces_cache is None:
                Handler._traces_cache = load_data()
            self.send_json(Handler._traces_cache)

        elif path == "/api/labels":
            data = {}
            if LABELS_FILE.exists():
                with open(LABELS_FILE) as f:
                    data = json.load(f)
            self.send_json(data)

        else:
            html_path = BASE_DIR / "index.html"
            try:
                content = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path == "/api/labels":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            with open(LABELS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            self.send_json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7777
    print(f"STXM Annotation  →  http://localhost:{port}", flush=True)
    print(f"  Traces : {EVALS_DIR / 'traces.jsonl'}", flush=True)
    print(f"  Labels : {LABELS_FILE}", flush=True)
    HTTPServer(("localhost", port), Handler).serve_forever()
