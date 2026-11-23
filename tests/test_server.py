"""
Tests for KeySense REST API handler logic using in-memory mock request/response streams.
Ensures full test coverage without requiring network socket permissions.
"""

import io
import json
import unittest
from keyword_extractor.server import KeywordAPIHandler


class MockSocket:
    """Mock socket object wrapping in-memory BytesIO streams."""
    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "b" in mode:
            if "r" in mode:
                return self.rfile
            elif "w" in mode:
                return self.wfile
        raise ValueError(f"Unsupported mode: {mode}")

    def sendall(self, data):
        self.wfile.write(data)


def simulate_request(method: str, path: str, body: dict = None) -> tuple:
    """Simulate an HTTP request to KeywordAPIHandler and return (status, headers, body_json)."""
    body_bytes = json.dumps(body).encode("utf-8") if body else b""
    raw_req = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: localhost\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"\r\n"
    ).encode("utf-8") + body_bytes

    sock = MockSocket(raw_req)

    # Instantiate handler with mock client address and mock server
    class MockServer:
        def __init__(self):
            pass

    handler = KeywordAPIHandler(sock, ("127.0.0.1", 12345), MockServer())
    response_bytes = sock.wfile.getvalue()

    # Parse response
    header_end = response_bytes.find(b"\r\n\r\n")
    headers_part = response_bytes[:header_end].decode("utf-8", errors="replace")
    body_part = response_bytes[header_end + 4 :]

    status_line = headers_part.split("\r\n")[0]
    status_code = int(status_line.split()[1])

    data = None
    try:
        data = json.loads(body_part.decode("utf-8"))
    except Exception:
        data = body_part.decode("utf-8", errors="replace")

    return status_code, data


class TestServerHandler(unittest.TestCase):

    def test_vercel_handler_import(self):
        from api.index import handler
        self.assertTrue(issubclass(handler, KeywordAPIHandler))

    def test_health_endpoint(self):
        status, data = simulate_request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "ok")

    def test_methods_endpoint(self):
        status, data = simulate_request("GET", "/api/methods")
        self.assertEqual(status, 200)
        self.assertIn("methods", data)
        self.assertIn("ensemble", data["methods"])

    def test_extract_endpoint(self):
        payload = {
            "text": "Deep reinforcement learning agents learn optimal decision policies.",
            "top_k": 3,
            "method": "ensemble"
        }
        status, data = simulate_request("POST", "/api/extract", payload)
        self.assertEqual(status, 200)
        self.assertIn("keywords", data)
        self.assertGreater(len(data["keywords"]), 0)

    def test_invalid_path(self):
        status, data = simulate_request("GET", "/api/unknown_endpoint")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
