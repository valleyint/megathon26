"""High-performance zero-dependency HTTP server for Stage 1 & Stage 2 Frontend."""

import json
import os
import sys
import mimetypes
import argparse
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Ensure megathon root in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.web.engine import CommonsStudioEngine

# Static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
engine = CommonsStudioEngine()


class StudioHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for CommonsCat AI Studio."""

    server_version = "CommonsCatStudio/1.0"

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400):
        self._send_json({"error": message, "status": status}, status=status)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handles GET requests for APIs and static web assets."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({
                "status": "online",
                "system": "Wikimedia Commons Category Suggestion Engine",
                "stages": ["Stage 1: Dense Multi-Scorer Recall", "Stage 2: Metadata & DAG Pruning"],
                "categories_count": len(engine.vocab),
                "dag_edges_count": len(engine.dag.edges),
            })
            return

        if path == "/api/presets":
            self._send_json({"presets": engine.get_presets()})
            return

        if path == "/api/dag":
            qs = parse_qs(parsed.query)
            active_cats = qs.get("categories", [])
            if active_cats and "," in active_cats[0]:
                active_cats = [c.strip() for c in active_cats[0].split(",") if c.strip()]
            dag_data = engine.get_dag_graph(active_categories=active_cats)
            self._send_json(dag_data)
            return

        # Serve static assets
        if path == "/" or path == "":
            file_path = STATIC_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            file_path = (STATIC_DIR / rel).resolve()
            # Prevent directory traversal
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self._send_error_json("Forbidden", 403)
                return

        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self._send_error_json(str(e), 500)
        else:
            # Fallback to index.html for SPA routes
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_error_json("File not found", 404)

    def do_POST(self):
        """Handles POST requests for Stage 1, Stage 2, and full pipeline runs."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            payload = json.loads(raw_body) if raw_body.strip() else {}
        except Exception as e:
            self._send_error_json(f"Malformed JSON payload: {e}", 400)
            return

        query_data = payload.get("query", {})
        params = payload.get("params", {})

        if path == "/api/stage1":
            try:
                res = engine.run_stage1(query_data=query_data, params=params)
                self._send_json(res)
            except Exception as e:
                self._send_error_json(f"Stage 1 execution error: {e}", 500)
            return

        if path == "/api/stage2":
            try:
                stage1_pool = payload.get("stage1_pool", {})
                res = engine.run_stage2(stage1_pool_dict=stage1_pool, query_data=query_data, params=params)
                self._send_json(res)
            except Exception as e:
                self._send_error_json(f"Stage 2 execution error: {e}", 500)
            return

        if path == "/api/pipeline":
            try:
                res = engine.run_pipeline(query_data=query_data, params=params)
                self._send_json(res)
            except Exception as e:
                self._send_error_json(f"Pipeline execution error: {e}", 500)
            return

        self._send_error_json(f"Unknown endpoint: {path}", 404)


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    """Starts the HTTP server."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, StudioHTTPRequestHandler)
    print(f"============================================================")
    print(f"CommonsCat AI Studio (Stage 1 & Stage 2) Server Running!")
    print(f"Local Access:   http://localhost:{port}")
    print(f"Network Access: http://{host}:{port}")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CommonsCat AI Frontend Studio Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
