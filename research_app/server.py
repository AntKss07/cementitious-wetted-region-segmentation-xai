"""Local, read-only HTTP server. Run: python -m research_app.server."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .catalog import Catalog

STATIC = Path(__file__).parent / "static"


def handler_for(catalog):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type, status=200, download=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if download:
                safe = ''.join(c for c in download if c.isascii() and (c.isalnum() or c in '._- '))
                self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status=200):
            self.send(json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            get = lambda key, default="": query.get(key, [default])[0]
            try:
                if url.path == "/api/meta":
                    self.send_json(catalog.metadata())
                elif url.path == "/api/frame":
                    self.send_json(catalog.frame(get("video"), get("stem")))
                elif url.path == "/api/panel":
                    self.send(catalog.panel(get("model"), get("video"), get("stem"), get("view", "prediction"), get("mode", "raw")), "image/png")
                elif url.path == "/api/agreement":
                    self.send(catalog.agreement(get("video"), get("stem")), "image/png")
                elif url.path == "/api/comparison":
                    self.send(catalog.comparison(get("video"), get("stem"), get("mode", "raw")), "image/png", download="segmentation-comparison.png")
                elif url.path.startswith("/api/files/"):
                    path = catalog.path_for_id(url.path.rsplit("/", 1)[-1])
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self.send(path.read_bytes(), mime, download=path.name if get("download") == "1" or path.suffix in {".ipynb", ".md"} else None)
                elif url.path in {"/", "/index.html", "/app.js", "/state.mjs", "/styles.css"}:
                    path = STATIC / ("index.html" if url.path == "/" else url.path[1:])
                    self.send(path.read_bytes(), {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}[path.suffix])
                else:
                    self.send_json({"error": "Not found"}, 404)
            except KeyError as exc:
                self.send_json({"error": str(exc).strip("'")}, 404)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            except (OSError, RuntimeError):
                self.send_json({"error": "Artifact cannot be read. It may be missing or malformed."}, 422)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            # Avoid logging local data paths or query values by default.
            if os.getenv("RESEARCH_HTTP_LOG") == "1":
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.getenv("RESEARCH_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(Catalog(args.root)))
    print(f"Research explorer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
