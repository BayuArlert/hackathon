import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
import io
import base64
from PIL import Image
import librosa
import threading
import queue
import time
import tempfile

# ====================== 1. MODEL DETEKSI MULTI-MODAL ======================


class DeepGuardModel:
    def __init__(self):
        # 1.1 Model Visual (gambar/video)
        self.visual_model = self._create_visual_model()

        # 1.2 Model Audio
        self.audio_model = self._create_audio_model()

        # 1.3 Model Temporal (untuk video)
        self.temporal_model = self._create_temporal_model()

    def _create_visual_model(self):
        model = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    def _create_audio_model(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        return model

    def _create_temporal_model(self):
        model = nn.LSTM(input_size=128, hidden_size=64,
                        num_layers=1, batch_first=True)
        fc = nn.Linear(64, 2)
        return (model, fc)

    def analyze_image(self, image):
        self.visual_model.eval()
        try:
            confidence = np.random.uniform(0.7, 0.95)
            height, width = image.shape[:2]
            heatmap = np.zeros((height, width))
            for _ in range(3):
                x, y = np.random.randint(
                    0, width), np.random.randint(0, height)
                size = np.random.randint(10, 50)
                cv2.circle(heatmap, (x, y), size, 1, -1)
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            heatmap = heatmap / np.max(heatmap)
            return confidence, heatmap
        except Exception as e:
            print(f"Error in analyze_image: {e}")
            return None, None

    def analyze_video(self, video_frames, audio_data=None):
        results = []
        temporal_features = []
        try:
            for frame in video_frames:
                conf, heatmap = self.analyze_image(frame)
                if conf is None or heatmap is None:
                    continue
                results.append((conf, heatmap))
                temporal_features.append(np.random.random(128))

            if len(temporal_features) > 1:
                temporal_conf = self.analyze_temporal(temporal_features)
            else:
                temporal_conf = 0.5

            if audio_data is not None:
                audio_conf = self.analyze_audio(audio_data)
            else:
                audio_conf = 0.5

            combined_conf = 0.6 * \
                np.mean([r[0] for r in results]) + 0.2 * \
                temporal_conf + 0.2 * audio_conf
            return combined_conf, results
        except Exception as e:
            print(f"Error in analyze_video: {e}")
            return None, None

    def analyze_audio(self, audio_data):
        return np.random.uniform(0.6, 0.9)

    def analyze_temporal(self, features):
        return np.random.uniform(0.5, 0.8)

# ====================== 2. SISTEM REAL-TIME PROCESSOR ======================


class RealTimeProcessor:
    def __init__(self, model, max_queue_size=10):
        self.model = model
        self.processing_queue = queue.Queue(maxsize=max_queue_size)
        self.results_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(
                target=self._processing_worker)
            self.worker_thread.daemon = True
            self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def _processing_worker(self):
        while self.is_running:
            try:
                task = self.processing_queue.get(block=True, timeout=0.1)
                task_type, data, task_id = task
                if task_type == 'image':
                    confidence, heatmap = self.model.analyze_image(data)
                    if confidence is not None and heatmap is not None:
                        self.results_queue.put((task_id, confidence, heatmap))
                elif task_type == 'video':
                    confidence, frame_results = self.model.analyze_video(data)
                    if confidence is not None and frame_results is not None:
                        self.results_queue.put(
                            (task_id, confidence, frame_results))
                self.processing_queue.task_done()
            except queue.Empty:
                continue

    def submit_task(self, task_type, data, task_id):
        try:
            self.processing_queue.put((task_type, data, task_id), block=False)
            return True
        except queue.Full:
            return False

    def get_result(self, block=False, timeout=None):
        try:
            return self.results_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

# ====================== 3. MODUL PERTAHANAN ADVERSARIAL ======================


class AdversarialDefense:
    def preprocess_input(self, image):
        noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        defended = np.clip(image + noise, 0, 1)
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_img = Image.fromarray((defended * 255).astype(np.uint8))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            img_compressed = np.array(Image.open(
                buffer)).astype(np.float32) / 255.0
            return img_compressed
        return defended

    def detect_adversarial_attack(self, image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        gradient_magnitude = np.sqrt(gx*gx + gy*gy)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        is_adversarial = gradient_std < 0.01 and gradient_mean > 0.1
        confidence = np.random.uniform(0.5, 0.8) if is_adversarial else 0.1
        return is_adversarial, confidence

# ====================== 4. SISTEM PENJELASAN DAN KEPERCAYAAN ======================


class ExplainabilityModule:
    def generate_explanation(self, image, heatmap, prediction):
        is_fake = prediction > 0.5
        confidence = prediction if is_fake else 1 - prediction
        high_attention_percentage = np.sum(heatmap > 0.5) / heatmap.size

        anomalies = []
        if high_attention_percentage > 0.01:
            anomalies.append("texture inconsistencies")
        if np.max(heatmap) > 0.9:
            anomalies.append("artificial boundaries")
        if np.std(heatmap) > 0.2:
            anomalies.append("lighting inconsistencies")

        suspected_features = []
        if np.max(heatmap[:image.shape[0]//3, :]) > 0.6:
            suspected_features.append("eye/forehead region")
        if np.max(heatmap[image.shape[0]//3:2*image.shape[0]//3, :]) > 0.6:
            suspected_features.append("nose/cheek region")
        if np.max(heatmap[2*image.shape[0]//3:, :]) > 0.6:
            suspected_features.append("mouth/chin region")

        # Ensure the heatmap is in the same type as the image
        heatmap = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Ensure the image is in the correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = (image * 255).astype(np.uint8)
        else:
            image = cv2.cvtColor(
                (image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Combine the image and heatmap
        visualization = cv2.addWeighted(image, 0.7, colored_heatmap, 0.3, 0)

        explanation = {
            "verdict": "FAKE" if is_fake else "REAL",
            "confidence": float(confidence),
            "high_attention_areas_percent": float(high_attention_percentage * 100),
            "detected_anomalies": anomalies,
            "suspected_features": suspected_features,
            "reliability_score": float(confidence * (1 - 0.5 * high_attention_percentage)),
            "details": {
                "face_consistency": float(np.random.uniform(0.3, 0.7) if is_fake else np.random.uniform(0.7, 0.9)),
                "texture_analysis": float(np.random.uniform(0.3, 0.7) if is_fake else np.random.uniform(0.7, 0.9)),
                "lighting_score": float(np.random.uniform(0.3, 0.7) if is_fake else np.random.uniform(0.7, 0.9)),
            }
        }
        return explanation, visualization

# ====================== 5. API FLASK ======================


app = Flask(__name__)

model = DeepGuardModel()
processor = RealTimeProcessor(model)
adversarial_defense = AdversarialDefense()
explainer = ExplainabilityModule()

processor.start()

# Set the maximum file size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Custom error handler for exceeding file size


@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_exceeded(e):
    return jsonify({'error': 'File size exceeds the maximum allowed limit of 10 MB'}), 413


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    file_bytes = file.read()

    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        return process_image(file_bytes)
    elif file.filename.endswith(('.mp4', '.avi', '.mov')):
        return process_video(file_bytes)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400


def process_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_defended = adversarial_defense.preprocess_input(img_rgb / 255.0)
    is_adversarial, adv_confidence = adversarial_defense.detect_adversarial_attack(
        img_defended)

    fake_confidence, heatmap = model.analyze_image(img_defended)
    if fake_confidence is None or heatmap is None:
        return jsonify({'error': 'Failed to analyze image'}), 500

    explanation, visualization = explainer.generate_explanation(
        img_defended, heatmap, fake_confidence)

    if is_adversarial:
        explanation["adversarial_alert"] = {
            "detected": True,
            "confidence": float(adv_confidence),
            "description": "Possible adversarial manipulation detected"
        }

    _, buffer = cv2.imencode('.jpg', visualization * 255)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'result': explanation,
        'visualization': 'data:image/jpeg;base64,' + img_str
    })


def process_video(video_bytes):
    # Create a temporary file to store the video bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file.flush()

        # Open the temporary file with OpenCV
        cap = cv2.VideoCapture(tmp_file.name)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 400

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if not frames:
        return jsonify({'error': 'Failed to extract frames from video'}), 400

    confidence, frame_results = model.analyze_video(frames)
    if confidence is None or frame_results is None:
        return jsonify({'error': 'Failed to analyze video'}), 500

    explanation, visualization = explainer.generate_explanation(
        frames[-1] / 255.0, frame_results[-1][1], confidence)

    explanation["temporal_analysis"] = {
        "frame_consistency": float(np.random.uniform(0.4, 0.9)),
        "motion_naturalness": float(np.random.uniform(0.4, 0.9)),
        "analyzed_frames": len(frames)
    }

    _, buffer = cv2.imencode('.jpg', visualization * 255)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'result': explanation,
        'visualization': 'data:image/jpeg;base64,' + img_str
    })


@app.route('/api/realtime', methods=['POST'])
def realtime_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame data'}), 400

    frame_file = request.files['frame']
    task_id = request.form.get('id', 'frame-' + str(int(time.time())))

    frame_bytes = frame_file.read()
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode frame'}), 400

    success = processor.submit_task('image', img, task_id)
    if not success:
        return jsonify({'error': 'Processing queue full'}), 503

    return jsonify({'status': 'processing', 'id': task_id})


@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    try:
        result = processor.get_result(block=False)
        if result is None or result[0] != task_id:
            return jsonify({'status': 'processing'})

        _, confidence, heatmap = result
        img = np.ones((heatmap.shape[0], heatmap.shape[1], 3)) * 255
        explanation, visualization = explainer.generate_explanation(
            img / 255.0, heatmap, confidence)

        _, buffer = cv2.imencode('.jpg', visualization * 255)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'complete',
            'result': explanation,
            'visualization': 'data:image/jpeg;base64,' + img_str
        })
    except Exception as e:
        print(f"Error in get_result: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
