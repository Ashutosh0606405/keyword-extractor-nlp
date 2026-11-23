"""
Vercel Serverless Function entry point for KeySense NLP.
Exposes 'handler' subclassing BaseHTTPRequestHandler for @vercel/python.
"""

import sys
import os
from urllib.parse import urlparse, parse_qs

# Ensure package root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keyword_extractor.server import KeywordAPIHandler


class handler(KeywordAPIHandler):
    """
    Vercel serverless HTTP request handler.
    Normalizes Vercel rewrites and header paths before dispatching.
    """

    def _resolve_path(self) -> str:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        # 1. Check rewrite query param
        if "__path__" in qs:
            target = qs["__path__"][0].strip()
            if not target.startswith("/"):
                target = "/" + target
            if not target.startswith("/api"):
                target = "/api" + target
            return target

        # 2. Check standard Vercel proxy headers
        for h in ("x-matched-path", "x-vercel-matched-path", "x-forwarded-uri", "x-original-uri"):
            val = self.headers.get(h)
            if val:
                return urlparse(val).path

        return parsed.path

    def do_GET(self):
        self.path = self._resolve_path()
        return super().do_GET()

    def do_POST(self):
        self.path = self._resolve_path()
        return super().do_POST()

    def do_OPTIONS(self):
        self.path = self._resolve_path()
        return super().do_OPTIONS()
