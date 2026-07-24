"""
前端静态文件服务器
用法: python run_frontend.py [端口]
默认端口: 3000
"""
import http.server
import socketserver
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # 禁止缓存，方便开发调试
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[Frontend] {args[0]}")


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"前端服务已启动: http://localhost:{PORT}")
    print(f"后端 API 地址: http://127.0.0.1:8001")
    print("按 Ctrl+C 停止服务")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n前端服务已停止")
