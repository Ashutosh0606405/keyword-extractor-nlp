"""
REST API and Static Web Server for Keyword Extractor NLP.
Zero-dependency native HTTP server with JSON REST API and web UI serving,
plus optional FastAPI application export when fastapi is installed.
"""

import json
import os
import sys
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from keyword_extractor.pipeline import KeywordExtractorPipeline
from keyword_extractor.evaluation import run_benchmark, BENCHMARK_DATASET


# Global Pipeline instance
_PIPELINE = KeywordExtractorPipeline()

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "static")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for handling concurrent requests."""
    daemon_threads = True


class KeywordAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing REST API endpoints and static file serving."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, "text/html; charset=utf-8")
                self.wfile.write(content)
            else:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"Static web UI not found.")
            return

        if path == "/api/health":
            self._set_headers(200)
            resp = {"status": "ok", "version": "1.0.0", "service": "KeySense NLP"}
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path == "/api/methods":
            self._set_headers(200)
            resp = {
                "methods": sorted(list(KeywordExtractorPipeline.AVAILABLE_METHODS)),
                "default": _PIPELINE.method_name
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path == "/api/benchmark":
            try:
                results = run_benchmark(_PIPELINE)
                self._set_headers(200)
                self.wfile.write(json.dumps({"benchmark": results}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode("utf-8"))
            return

        if path == "/api/extract":
            text = data.get("text", "")
            if not text:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "'text' field is required"}).encode("utf-8"))
                return

            method = data.get("method", "ensemble")
            top_k = int(data.get("top_k", 10))
            ngram_range = tuple(data.get("ngram_range", [1, 2]))
            diversity = float(data.get("diversity", 0.5))

            try:
                keywords = _PIPELINE.extract(
                    text=text,
                    top_k=top_k,
                    ngram_range=ngram_range,
                    method=method,
                    diversity=diversity
                )
                self._set_headers(200)
                resp = {
                    "method": method,
                    "count": len(keywords),
                    "keywords": [k.to_dict() for k in keywords]
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if path == "/api/compare":
            text = data.get("text", "")
            top_k = int(data.get("top_k", 5))
            ngram_range = tuple(data.get("ngram_range", [1, 2]))

            try:
                comp = _PIPELINE.compare(text=text, top_k=top_k, ngram_range=ngram_range)
                resp = {m: [k.to_dict() for k in res] for m, res in comp.items()}
                self._set_headers(200)
                self.wfile.write(json.dumps({"comparison": resp}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if path == "/api/batch":
            texts = data.get("texts", [])
            method = data.get("method", "ensemble")
            top_k = int(data.get("top_k", 5))

            try:
                batch_res = _PIPELINE.extract_batch(texts, top_k=top_k, method=method)
                resp = {
                    "total_documents": len(texts),
                    "results": [[k.to_dict() for k in doc_res] for doc_res in batch_res]
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the multi-threaded HTTP/REST server."""
    server = ThreadedHTTPServer((host, port), KeywordAPIHandler)
    print(f"\n==========================================================")
    print(f" KeySense NLP API & Web UI running at: http://localhost:{port}")
    print(f" Endpoints:")
    print(f"   - GET  /              (Interactive Web UI)")
    print(f"   - GET  /api/health    (Service Health)")
    print(f"   - GET  /api/methods   (List Algorithms)")
    print(f"   - POST /api/extract   (Single Document Extraction)")
    print(f"   - POST /api/compare   (Multi-Method Comparison)")
    print(f"   - POST /api/batch     (Batch Extraction)")
    print(f"   - GET  /api/benchmark (Evaluation Benchmarks)")
    print(f"==========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start KeySense NLP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding")
    parser.add_argument("--port", type=int, default=8000, help="Port binding")
    args = parser.parse_args()
    start_server(args.host, args.port)
