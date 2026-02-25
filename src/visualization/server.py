"""Simple HTTP server for dashboard viewing.

Uses stdlib http.server to serve the generated dashboard HTML and static JS modules.
"""

from __future__ import annotations

import http.server
import logging
import mimetypes
import socketserver
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


class DashboardServer:
    """Lightweight HTTP server for viewing dashboard."""

    def __init__(self, dashboard_path: str, port: int = 8000):
        """Initialize server.

        Args:
            dashboard_path: Path to the dashboard HTML file
            port: Port to serve on (default: 8000)
        """
        self.dashboard_path = Path(dashboard_path)
        self.port = port

        if not self.dashboard_path.exists():
            raise FileNotFoundError(f"Dashboard file not found: {dashboard_path}")

    def serve(self, open_browser: bool = True) -> None:
        """Start server and optionally open browser.

        Args:
            open_browser: If True, opens browser automatically (default: True)
        """
        # Serve from visualization root to access both templates/ and static/
        directory = str(Path(__file__).parent.absolute())

        # Ensure JavaScript modules have correct MIME type
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("application/javascript", ".mjs")

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, format, *args):
                """Override to suppress request logging noise."""
                logger.debug(format % args)

        with socketserver.TCPServer(("", self.port), Handler) as httpd:
            # Construct URL relative to visualization root
            rel_path = self.dashboard_path.relative_to(Path(__file__).parent)
            url = f"http://localhost:{self.port}/{rel_path.as_posix()}"
            logger.info(f"Serving dashboard at {url}")
            print(f"Dashboard available at: {url}")
            print("Press Ctrl+C to stop server")

            if open_browser:
                webbrowser.open(url)

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped")


def start_server(dashboard_path: str, port: int = 8000, open_browser: bool = True) -> None:
    """Convenience function to start dashboard server.

    Args:
        dashboard_path: Path to the dashboard HTML file
        port: Port to serve on (default: 8000)
        open_browser: If True, opens browser automatically (default: True)
    """
    server = DashboardServer(dashboard_path, port)
    server.serve(open_browser)
