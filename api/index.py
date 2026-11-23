"""
Vercel Serverless Function entry point for KeySense NLP.
Exposes 'handler' subclassing BaseHTTPRequestHandler for @vercel/python.
"""

import sys
import os

# Ensure package root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keyword_extractor.server import KeywordAPIHandler


class handler(KeywordAPIHandler):
    """
    Vercel serverless HTTP request handler.
    Inherits all REST endpoints from KeywordAPIHandler:
      - GET  /api/health
      - GET  /api/methods
      - GET  /api/benchmark
      - POST /api/extract
      - POST /api/compare
      - POST /api/batch
    """
    pass
