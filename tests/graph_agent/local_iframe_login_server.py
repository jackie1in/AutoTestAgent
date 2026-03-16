"""Local test server: login redirects to index that contains an iframe."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs


HOST = "127.0.0.1"
PORT = 18080
LOGIN_PATH = "/login"
INDEX_PATH = "/index"
IFRAME_PATH = "/embedded"
COOKIE_NAME = "session"
COOKIE_VALUE = "ok"


def _html(body: str, *, title: str) -> bytes:
    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    ).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    def _send_html(self, status: int, body: str, *, title: str) -> None:
        payload = _html(body, title=title)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str, *, with_session_cookie: bool = False) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        if with_session_cookie:
            self.send_header("Set-Cookie", f"{COOKIE_NAME}={COOKIE_VALUE}; Path=/")
        self.end_headers()

    def _is_logged_in(self) -> bool:
        cookie = self.headers.get("Cookie", "")
        return f"{COOKIE_NAME}={COOKIE_VALUE}" in cookie

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", LOGIN_PATH}:
            self._send_html(
                HTTPStatus.OK,
                (
                    "<h1>Login Page</h1>"
                    "<form method='post' action='/do-login'>"
                    "<label for='username'>Username</label>"
                    "<input id='username' name='username' />"
                    "<label for='password'>Password</label>"
                    "<input id='password' name='password' type='password' />"
                    "<button id='submit' type='submit'>Login</button>"
                    "</form>"
                ),
                title="Login",
            )
            return

        if self.path == INDEX_PATH:
            if not self._is_logged_in():
                self._redirect(LOGIN_PATH)
                return
            self._send_html(
                HTTPStatus.OK,
                (
                    "<h1>Index Page</h1>"
                    "<p id='welcome'>Welcome!</p>"
                    "<iframe id='content-frame' src='/embedded' "
                    "title='embedded frame'></iframe>"
                ),
                title="Index",
            )
            return

        if self.path == IFRAME_PATH:
            self._send_html(
                HTTPStatus.OK,
                (
                    "<h2>Embedded Frame</h2>"
                    "<button id='inside-frame-btn'>Inside Frame Button</button>"
                ),
                title="Embedded",
            )
            return

        self._send_html(
            HTTPStatus.NOT_FOUND,
            "<h1>404 Not Found</h1>",
            title="Not Found",
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/do-login":
            self._send_html(
                HTTPStatus.NOT_FOUND,
                "<h1>404 Not Found</h1>",
                title="Not Found",
            )
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw)
        username = (data.get("username") or [""])[0]
        password = (data.get("password") or [""])[0]

        if username == "demo" and password == "demo":
            self._redirect(INDEX_PATH, with_session_cookie=True)
            return

        self._send_html(
            HTTPStatus.UNAUTHORIZED,
            (
                "<h1>Login Failed</h1>"
                "<p id='error'>Invalid credentials</p>"
                "<a href='/login' id='back-login'>Back to login</a>"
            ),
            title="Login Failed",
        )

    def log_message(self, format: str, *args: object) -> None:
        # Keep terminal output clean while mapping runs.
        return


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local login->index(with iframe) test server."
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"Bind host (default: {HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Bind port (default: {PORT})",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"Test server listening on http://{args.host}:{args.port}/login")
    print("Credential: demo / demo")
    server.serve_forever()


if __name__ == "__main__":
    main()
