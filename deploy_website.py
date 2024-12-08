import http.server
import socketserver


PORT = 8088
SRC_DIR = "src"


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SRC_DIR)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving Front-end at http://localhost:{PORT}")
        httpd.serve_forever()