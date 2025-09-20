#!/usr/bin/env python3
"""
Simple HTTP server for the UrbanSense web demo
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve the web demo with proper MIME types"""

    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def guess_type(self, path):
        """Override to handle additional file types"""
        mimetype = super().guess_type(path)
        if path.endswith('.js'):
            return 'application/javascript'
        elif path.endswith('.css'):
            return 'text/css'
        return mimetype

def start_server(port=8080):
    """Start the web demo server"""
    # Change to web_demo directory
    web_demo_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_demo_path)

    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            print(f"🌐 UrbanSense Web Demo Server Starting...")
            print(f"📍 Server running at: http://localhost:{port}")
            print(f"📁 Serving from: {web_demo_path}")
            print(f"🚀 Opening browser automatically...")
            print("=" * 60)

            # Open browser after a short delay
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(f'http://localhost:{port}')

            browser_thread = threading.Thread(target=open_browser, daemon=True)
            browser_thread.start()

            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            start_server(port + 1)
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8080.")

    start_server(port)