from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2, numpy as np, os, time, uuid, threading, base64, subprocess, requests

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Directory Setup
# -----------------------------
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load YOLOv3
# -----------------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names") as f:
    classes = [line.strip() for line in f]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# -----------------------------
# TomTom Traffic Configuration
# -----------------------------
TOMTOM_API_KEY = "1vo0s7uUAWLbKwsfQY4IherNlc0hpKGk"
BENGALURU_BBOX = {
    "minLon": 77.4500,
    "minLat": 12.8500,
    "maxLon": 77.7500,
    "maxLat": 13.1000
}


# -----------------------------
# Video Processing Function
# -----------------------------
def process_video(socket_id, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for out_ in outs:
            for detection in out_:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(detection[0] * width - w / 2)
                    y = int(detection[1] * height - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i] % len(colors)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

        if frame_idx % 10 == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            socketio.emit("frame", {
                "sid": socket_id,
                "frame": base64.b64encode(buffer).decode("utf-8")
            })

        progress = (frame_idx / total_frames) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (progress / 100)) - elapsed if progress > 1 else 0
        if frame_idx % 5 == 0:
            socketio.emit("progress", {
                "sid": socket_id,
                "progress": round(progress, 2),
                "eta": round(eta, 1)
            })

    cap.release()
    out.release()

    ffmpeg = "bin/ffmpeg.exe"
    if os.path.exists(ffmpeg):
        subprocess.run([
            ffmpeg, "-y", "-i", temp_out,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_out)
    else:
        os.rename(temp_out, output_path)

    socketio.emit("complete", {
        "sid": socket_id,
        "video_url": f"/{output_path}"
    })


# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/video")
def video():
    return render_template("index.html")


@app.route("/predictions")
def predictions():
    return render_template("predictions.html")


@app.route("/heatmap")
def heatmap():
    return render_template("heatmap.html")


# -----------------------------
# TomTom API Proxy Route
# -----------------------------
@app.route("/api/incidents")
def get_incidents():
    bbox = f"{BENGALURU_BBOX['minLon']},{BENGALURU_BBOX['minLat']},{BENGALURU_BBOX['maxLon']},{BENGALURU_BBOX['maxLat']}"
    url = f"https://api.tomtom.com/traffic/services/5/incidentDetails?bbox={bbox}&key={TOMTOM_API_KEY}&language=en-GB&timeValidityFilter=present"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Heatmap Data API
# -----------------------------
@app.route("/api/heatmap")
def get_heatmap():
    """Returns top 10 congested areas with percentage"""
    location_query = request.args.get("location", "").lower()

    bbox = f"{BENGALURU_BBOX['minLon']},{BENGALURU_BBOX['minLat']},{BENGALURU_BBOX['maxLon']},{BENGALURU_BBOX['maxLat']}"
    url = f"https://api.tomtom.com/traffic/services/5/incidentDetails?bbox={bbox}&key={TOMTOM_API_KEY}&language=en-GB&timeValidityFilter=present"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        incidents = data.get("incidents", [])
        area_count = {}
        for inc in incidents:
            area = inc["properties"].get("from", "Unknown Area")
            area_count[area] = area_count.get(area, 0) + 1

        sorted_areas = sorted(area_count.items(), key=lambda x: x[1], reverse=True)

        if location_query:
            sorted_areas = [a for a in sorted_areas if location_query in a[0].lower()]

        top_areas = sorted_areas[:10]
        total_incidents = sum(area_count.values()) or 1

        response = []
        for name, count in top_areas:
            percent = round((count / total_incidents) * 100, 2)
            response.append({"area": name, "incidents": count, "percent": percent})

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Upload Video Endpoint
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    filename = f"{uuid.uuid4().hex}.mp4"
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")
    file.save(input_path)

    socket_id = request.form.get("sid")
    threading.Thread(target=process_video, args=(socket_id, input_path, output_path)).start()
    return {"status": "processing"}


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    socketio.run(app, debug=True)
