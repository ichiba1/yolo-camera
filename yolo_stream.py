"""カメラ映像に YOLO をかけて、検出結果をブラウザに配信する。

起動: ~/yolo-env/bin/python ~/yolo-camera/yolo_stream.py
閲覧: パソコンで ssh -L 8000:localhost:8000 <ユーザー名>@<ラズパイのアドレス> → http://localhost:8000
"""
import threading
import time
from http import server
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

PORT = 8000
MODEL = Path(__file__).with_name("yolo11n_ncnn_model")

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>YOLO Camera</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream.mjpg"></body></html>"""


class LatestFrame:
    def __init__(self):
        self.jpeg = None
        self.condition = threading.Condition()

    def put(self, jpeg):
        with self.condition:
            self.jpeg = jpeg
            self.condition.notify_all()


latest = LatestFrame()


def detect_loop():
    model = YOLO(str(MODEL), task="detect")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()
    fps = 0.0
    while True:
        t0 = time.time()
        frame = picam2.capture_array()  # RGB888 は OpenCV と同じ BGR 並び
        result = model(frame, verbose=False)[0]
        annotated = result.plot()
        fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
        cv2.putText(annotated, f"{fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            latest.put(buf.tobytes())


class Handler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(PAGE))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with latest.condition:
                        latest.condition.wait()
                        jpeg = latest.jpeg
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


class Server(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    threading.Thread(target=detect_loop, daemon=True).start()
    print(f"streaming on http://127.0.0.1:{PORT}", flush=True)
    Server(("127.0.0.1", PORT), Handler).serve_forever()
