import sys
import os
import http
from http.server import BaseHTTPRequestHandler, HTTPServer

import whats_see

PORT = 8888

whats_see.current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))


class WhatsSeeHandler(BaseHTTPRequestHandler):

    # Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(bytes(whats_see.current_work_dir, "UTF-8"))
        return


if __name__ == "__main__":

    try:

        server = HTTPServer(('', PORT), WhatsSeeHandler)
        print('Started WhatsSee SERVER on port ', PORT)

        server.serve_forever()

    except KeyboardInterrupt:
        server.socket.close()
