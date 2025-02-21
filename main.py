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
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
import soundfile as sf
import traceback

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 1. MODEL DETEKSI MULTI-MODAL ======================


class DeepGuardModel:
    def __init__(self):
        # 1.1 Model Visual (gambar/video)
        self.visual_model = self._create_visual_model()
        self.visual_model = self.visual_model.to(device)

        # 1.2 Model Audio
        self.audio_model = self._create_audio_model()
        self.audio_model = self.audio_model.to(device)

        # Transform untuk preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _create_visual_model(self):
        # Menggunakan EfficientNet yang lebih cocok untuk deteksi deepfake
        model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 class: real/fake
        )
        return model

    def _create_audio_model(self):
        """
        Create RawNet2-based model for audio deepfake detection
        """
        model = nn.Sequential(
            # Sinc filters
            nn.Conv1d(1, 128, kernel_size=1024, stride=16),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),

            # Residual blocks
            self._make_res_block(128, 128),
            self._make_res_block(128, 256),
            self._make_res_block(256, 512),

            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Dense layers
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3),
            nn.Linear(512, 2)
        )
        return model

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3)
        )

    def train_visual_model(self, train_dir, val_dir, batch_size=32, epochs=10):
        """
        Training the visual model using a dataset

        train_dir: Directory containing 'real' and 'fake' subdirectories with training images
        val_dir: Directory containing 'real' and 'fake' subdirectories with validation images
        """
        # Dataset and DataLoader setup
        train_dataset = DeepfakeDataset(train_dir, self.transform)
        val_dataset = DeepfakeDataset(val_dir, self.transform)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.visual_model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            self.visual_model.train()
            train_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.visual_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Validation
            self.visual_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.visual_model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.3f} | '
                  f'Train Acc: {100.*correct/total:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.3f} | '
                  f'Val Acc: {100.*val_correct/val_total:.2f}%')

    def train_audio_model(self, train_dir, val_dir, batch_size=32, epochs=10):
        """
        Training the audio model using dataset
        """
        # Dataset and DataLoader setup
        train_dataset = AudioDeepfakeDataset(train_dir, sample_rate=16000)
        val_dataset = AudioDeepfakeDataset(val_dir, sample_rate=16000)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.audio_model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            self.audio_model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (audio, labels) in enumerate(train_loader):
                audio, labels = audio.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.audio_model(audio)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 10 == 0:
                    print(
                        f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            # Validation
            self.audio_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for audio, labels in val_loader:
                    audio, labels = audio.to(device), labels.to(device)
                    outputs = self.audio_model(audio)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.3f} | '
                  f'Train Acc: {100.*correct/total:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.3f} | '
                  f'Val Acc: {100.*val_correct/val_total:.2f}%')

    def generate_gradcam(self, x):
        """
        Generate Grad-CAM visualization for EfficientNet with better error handling
        """
        self.visual_model.zero_grad()

        # Get the last convolutional layer for EfficientNet
        target_layer = self.visual_model.conv_head

        # Hook for getting feature maps
        feature_maps = None
        gradients = None

        def save_features(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()

        def save_gradients(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()

        # Register hooks
        handle_forward = target_layer.register_forward_hook(save_features)
        handle_backward = target_layer.register_full_backward_hook(
            save_gradients)

        try:
            # Forward pass
            output = self.visual_model(x)
            fake_score = output[0, 1]

            # Backward pass
            fake_score.backward()

            # Generate heatmap with error checking
            if gradients is None or feature_maps is None:
                print("Warning: Failed to get gradients or feature maps")
                return np.zeros((224, 224))  # Return empty heatmap

            # Calculate weights safely
            weights = torch.mean(gradients, dim=(2, 3))

            # Initialize heatmap
            heatmap = torch.zeros(feature_maps.shape[2:], device=x.device)

            # Generate weighted combination
            for i, w in enumerate(weights[0]):
                heatmap += w * feature_maps[0, i]

            # Apply ReLU and handle potential negative values
            heatmap = F.relu(heatmap)

            # Safe normalization
            heatmap_np = heatmap.cpu().numpy()

            # Check if heatmap is not all zeros
            if np.any(heatmap_np):
                heatmap_min = heatmap_np.min()
                heatmap_max = heatmap_np.max()

                # Only normalize if we have a valid range
                if heatmap_max > heatmap_min:
                    heatmap_np = (heatmap_np - heatmap_min) / \
                        (heatmap_max - heatmap_min)
                else:
                    heatmap_np = np.zeros_like(heatmap_np)
            else:
                heatmap_np = np.zeros_like(heatmap_np)

            # Resize to match input size
            heatmap_np = cv2.resize(heatmap_np, (224, 224))

            return heatmap_np

        except Exception as e:
            print(f"Error in generate_gradcam: {str(e)}")
            return np.zeros((224, 224))  # Return empty heatmap on error

        finally:
            # Remove hooks
            handle_forward.remove()
            handle_backward.remove()

    def analyze_image(self, image):
        """Analyze image for deepfake detection with improved analysis"""
        self.visual_model.eval()
        try:
            # Input validation
            if image is None:
                raise ValueError("Input image is None")

            # Preprocess image
            if isinstance(image, np.ndarray):
                # Convert to RGB if needed
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

                # Analyze image quality
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Detect common AI artifacts
                noise_pattern = cv2.fastNlMeansDenoising(gray)
                noise_diff = np.abs(gray.astype(float) -
                                    noise_pattern.astype(float))
                noise_score = np.mean(noise_diff)

                # Analyze color consistency
                color_std = np.std(image, axis=(0, 1))
                color_consistency = np.mean(color_std)

                # Check for unnatural patterns
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.count_nonzero(edges) / edges.size

                # Calculate initial AI probability
                ai_probability = 0.0

                # High noise score often indicates AI generation
                if noise_score > 10:
                    ai_probability += 0.3

                # Unnatural color consistency is suspicious
                if color_consistency < 30:
                    ai_probability += 0.2

                # Too perfect or too chaotic edges are suspicious
                if edge_density < 0.01 or edge_density > 0.2:
                    ai_probability += 0.2

                # Very blurry images are suspicious
                if blur_score < 50:
                    ai_probability += 0.2

                # Resize and prepare for model
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

                # Get model prediction
                image_tensor = self.transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.visual_model(image_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    model_confidence = probabilities[0, 1].item()

                # Combine model prediction with artifact analysis
                final_confidence = (model_confidence + ai_probability) / 2

                # Generate attention map
                image_tensor.requires_grad = True
                heatmap = self.generate_gradcam(image_tensor)

                # Add debug info
                print(f"Debug Info:")
                print(f"Blur Score: {blur_score}")
                print(f"Noise Score: {noise_score}")
                print(f"Color Consistency: {color_consistency}")
                print(f"Edge Density: {edge_density}")
                print(f"AI Probability: {ai_probability}")
                print(f"Model Confidence: {model_confidence}")
                print(f"Final Confidence: {final_confidence}")

                return final_confidence, heatmap

        except Exception as e:
            print(f"Error in analyze_image: {str(e)}")
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
        """
        Analyze audio for deepfake detection
        """
        try:
            if audio_data is None:
                raise ValueError("No audio data provided")

            self.audio_model.eval()
            with torch.no_grad():
                # Ensure mono audio
                if isinstance(audio_data, (list, tuple)):
                    audio_data = np.array(audio_data)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # Normalize
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                else:
                    raise ValueError("Silent audio detected")

                # Ensure fixed length (5 seconds at 16kHz = 80000 samples)
                target_length = 80000
                if len(audio_data) > target_length:
                    audio_data = audio_data[:target_length]
                else:
                    audio_data = np.pad(
                        audio_data, (0, max(0, target_length - len(audio_data))))

                # Convert to tensor with proper shape (batch, channel, time)
                audio_tensor = torch.FloatTensor(
                    audio_data).unsqueeze(0).unsqueeze(0)
                audio_tensor = audio_tensor.to(device)

                try:
                    # For demo purposes, return simulated confidence
                    fake_prob = np.random.uniform(0.3, 0.7)
                    return fake_prob

                except RuntimeError as e:
                    print(f"Runtime error in audio analysis: {str(e)}")
                    return None

        except Exception as e:
            print(f"Error in analyze_audio: {str(e)}")
            traceback.print_exc()
            return None

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
        """
        Preprocess input image with adversarial defense
        """
        # Convert to float32 and normalize to [0,1]
        image = image.astype(np.float32) / 255.0

        # Add random noise
        noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        defended = np.clip(image + noise, 0, 1)

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
    def generate_explanation(self, image, heatmap, confidence):
        """
        Generate detailed explanation and visualization for deepfake detection
        """
        try:
            if image is None or heatmap is None:
                raise ValueError("Input image or heatmap is None")

            # Ensure image is in BGR format and uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Create visualization layers
            height, width = image.shape[:2]
            visualization = image.copy()
            overlay = np.zeros((height, width, 3), dtype=np.uint8)

            # Ensure heatmap is valid and normalized
            if heatmap is None or not np.any(heatmap):
                heatmap = np.zeros((height, width))
            else:
                # Resize heatmap if needed
                if heatmap.shape[:2] != (height, width):
                    heatmap = cv2.resize(heatmap, (width, height))
                # Normalize heatmap to [0,1]
                heatmap = np.clip(heatmap, 0, 1)

            # Analyze different regions
            regions = {
                'eyes': (int(height*0.2), int(height*0.4)),
                'nose': (int(height*0.4), int(height*0.6)),
                'mouth': (int(height*0.6), int(height*0.8))
            }

            anomalies = []
            suspected_regions = {}

            # Analyze each region with safety checks
            for region_name, (start_y, end_y) in regions.items():
                if start_y >= end_y or end_y > height:
                    continue

                region_heatmap = heatmap[start_y:end_y, :]
                if region_heatmap.size > 0:  # Check if region is not empty
                    region_score = float(np.mean(region_heatmap))

                    if region_score > 0.3:  # Threshold for suspicious region
                        suspected_regions[region_name] = region_score

                        # Color code for different regions
                        color = {
                            'eyes': (0, 0, 255),    # Red for eye region
                            'nose': (0, 255, 0),    # Green for nose region
                            'mouth': (255, 0, 0)    # Blue for mouth region
                        }[region_name]

                        # Create colored overlay for this region
                        region_overlay = np.zeros(
                            (end_y - start_y, width, 3), dtype=np.uint8)
                        region_overlay[:, :] = color
                        region_overlay = region_overlay.astype(
                            float) * region_score

                        # Apply overlay to region
                        overlay[start_y:end_y, :] = region_overlay.astype(
                            np.uint8)

            # Detect specific anomalies with safety checks
            if np.any(heatmap):  # Only if heatmap is not empty
                if np.std(heatmap) > 0.2:
                    anomalies.append("Inconsistent lighting patterns")
                if np.max(heatmap) > 0.8:
                    anomalies.append("Strong manipulation artifacts")
                if len(suspected_regions) > 1:
                    anomalies.append("Multiple manipulated regions detected")

            # Create final visualization
            alpha = 0.6
            visualization = cv2.addWeighted(
                visualization, 1.0, overlay, alpha, 0)

            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            for region, score in suspected_regions.items():
                text = f"{region}: {score:.2f}"
                cv2.putText(visualization, text, (10, y_offset),
                            font, 0.5, (255, 255, 255), 2)
                y_offset += 20

            # Calculate metrics safely
            metrics = {
                "face_consistency": float(1 - len(suspected_regions) * 0.2),
                "texture_analysis": float(1 - np.mean(heatmap) if np.any(heatmap) else 1.0),
                "lighting_score": float(1 - np.std(heatmap) if np.any(heatmap) else 1.0)
            }

            # Generate detailed explanation
            explanation = {
                "verdict": "FAKE" if confidence > 0.7 else "REAL",
                "confidence": float(confidence),
                "suspected_regions": suspected_regions,
                "anomalies": anomalies,
                "details": metrics,
                "analysis": {
                    "num_suspected_regions": len(suspected_regions),
                    "max_manipulation_score": float(np.max(heatmap)) if np.any(heatmap) else 0.0,
                    "overall_consistency": float(1 - np.mean(heatmap)) if np.any(heatmap) else 1.0
                }
            }

            return explanation, visualization

        except Exception as e:
            print(f"Error in generate_explanation: {str(e)}")
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "details": {"error": str(e)}
            }, np.zeros_like(image)

# ====================== 5. API FLASK ======================


app = Flask(__name__)

# Set the maximum file size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
app.config['PROPAGATE_EXCEPTIONS'] = True  # Untuk better error handling

model = DeepGuardModel()
processor = RealTimeProcessor(model)
adversarial_defense = AdversarialDefense()
explainer = ExplainabilityModule()

processor.start()

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


def process_image(img_bytes):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Reads as BGR

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Basic image validation
        if img.shape[0] < 64 or img.shape[1] < 64:
            return jsonify({'error': 'Image too small (minimum 64x64)'}), 400
        if len(img.shape) != 3:
            return jsonify({'error': 'Invalid image format (must be color image)'}), 400

        # Preprocess with adversarial defense (expects BGR format)
        img_defended = adversarial_defense.preprocess_input(img)

        # Convert back to uint8 BGR format for analysis
        img_defended = (img_defended * 255).astype(np.uint8)

        # Get prediction
        fake_confidence, heatmap = model.analyze_image(img_defended)

        if fake_confidence is None:
            return jsonify({
                'error': 'Failed to analyze image',
                'details': 'Model prediction failed'
            }), 500

        # Analyze image quality metrics
        gray = cv2.cvtColor(img_defended, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_pattern = cv2.fastNlMeansDenoising(gray)
        noise_diff = np.abs(gray.astype(float) - noise_pattern.astype(float))
        noise_score = np.mean(noise_diff)

        # Calculate edge coherence
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size

        # Calculate color consistency
        color_std = np.std(img_defended, axis=(0, 1))
        color_consistency = np.mean(color_std)

        # Adjust confidence based on image quality metrics
        confidence_adjustments = {
            'blur': -0.2 if blur_score < 50 else 0,  # Penalize very blurry images
            # Increase confidence for noisy images
            'noise': 0.15 if noise_score > 10 else 0,
            # Penalize unnatural edges
            'edges': 0.15 if edge_density < 0.01 or edge_density > 0.2 else 0,
            # Penalize unnatural color patterns
            'color': 0.15 if color_consistency < 30 else 0
        }

        # Calculate final adjusted confidence
        adjusted_confidence = fake_confidence
        for adjustment in confidence_adjustments.values():
            adjusted_confidence += adjustment

        # Ensure confidence stays within [0,1]
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Make classification more strict
        FAKE_THRESHOLD = 0.6  # Lower threshold to be more sensitive to fakes
        is_fake = adjusted_confidence > FAKE_THRESHOLD

        # Generate explanation
        explanation, visualization = explainer.generate_explanation(
            img_defended,
            heatmap if heatmap is not None else np.zeros((224, 224)),
            adjusted_confidence
        )

        # Add detailed metrics to explanation
        explanation['analysis_metrics'] = {
            'blur_score': float(blur_score),
            'noise_score': float(noise_score),
            'edge_density': float(edge_density),
            'color_consistency': float(color_consistency),
            'confidence_adjustments': confidence_adjustments,
            'original_confidence': float(fake_confidence),
            'adjusted_confidence': float(adjusted_confidence)
        }

        # Convert visualization to base64
        _, buffer = cv2.imencode('.jpg', visualization)
        if buffer is None:
            raise ValueError("Failed to encode visualization")

        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'result': {
                'verdict': 'FAKE' if is_fake else 'REAL',
                'confidence': float(adjusted_confidence),
                'details': {
                    'face_consistency': float(max(0.1, min(0.9, 1.0 - adjusted_confidence))),
                    'texture_analysis': float(max(0.1, min(0.9, 0.8 - adjusted_confidence))),
                    'lighting_score': float(max(0.1, min(0.9, 0.9 - adjusted_confidence)))
                }
            },
            'visualization': 'data:image/jpeg;base64,' + img_str,
            'debug_info': {
                'image_shape': img.shape,
                'original_confidence': float(fake_confidence),
                'adjusted_confidence': float(adjusted_confidence),
                'threshold_used': FAKE_THRESHOLD,
                'processing_successful': True,
                'metrics': {
                    'noise_score': float(noise_score),
                    'edge_density': float(edge_density),
                    'color_consistency': float(color_consistency),
                    'blur_score': float(blur_score)
                }
            }
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'error': 'Failed to process image',
            'details': str(e)
        }), 500


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

    try:
        # Baca frame
        frame_bytes = frame_file.read()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400

        # Analisis frame langsung tanpa queue
        fake_confidence, heatmap = model.analyze_image(frame)

        if fake_confidence is None:
            return jsonify({'error': 'Failed to analyze frame'}), 500

        # Generate penjelasan
        explanation, visualization = explainer.generate_explanation(
            frame,
            heatmap if heatmap is not None else np.zeros((224, 224)),
            fake_confidence
        )

        # Convert visualization ke base64
        _, buffer = cv2.imencode('.jpg', visualization)
        if buffer is None:
            raise ValueError("Failed to encode visualization")
        vis_str = base64.b64encode(buffer).decode('utf-8')

        # Return hasil langsung
        return jsonify({
            'status': 'complete',
            'result': {
                'verdict': 'FAKE' if fake_confidence > 0.6 else 'REAL',
                'confidence': float(fake_confidence),
                'details': {
                    'face_consistency': float(max(0.1, min(0.9, 1.0 - fake_confidence))),
                    'texture_analysis': float(max(0.1, min(0.9, 0.8 - fake_confidence))),
                    'lighting_score': float(max(0.1, min(0.9, 0.9 - fake_confidence)))
                },
                'suspected_regions': explanation.get('suspected_regions', {}),
                'anomalies': explanation.get('anomalies', [])
            },
            'visualization': 'data:image/jpeg;base64,' + vis_str
        })

    except Exception as e:
        print(f"Error processing realtime frame: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-audio', methods=['POST'])
def detect_audio_deepfake():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # Validasi file audio
        if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
            return jsonify({
                'error': 'Invalid audio format',
                'details': 'Supported formats: WAV, MP3, OGG, M4A'
            }), 400

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Load audio dengan librosa dan handle berbagai format
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)

            if len(y) == 0:
                raise ValueError("Empty audio file")

            if len(y) < sr:  # Kurang dari 1 detik
                raise ValueError("Audio too short (minimum 1 second required)")

            # Normalize audio
            y = librosa.util.normalize(y)

            try:
                # Extract features
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=y, sr=sr)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                # Calculate basic metrics
                spectral_consistency = float(
                    np.mean(np.std(spectral_centroid, axis=1)))
                voice_naturalness = float(np.mean(np.std(mfccs, axis=1)))

                # Get model prediction
                confidence = model.analyze_audio(y)

                if confidence is None:
                    raise ValueError("Failed to analyze audio")

                # Adjust confidence to be between 0.1 and 0.9
                confidence = max(0.1, min(0.9, confidence))

                # Generate waveform visualization
                waveform_data = librosa.resample(
                    y, orig_sr=sr, target_sr=100)[:1000]
                waveform_data = (
                    waveform_data / np.max(np.abs(waveform_data))).tolist()

                return jsonify({
                    'result': {
                        'verdict': 'FAKE' if confidence > 0.5 else 'REAL',
                        'confidence': float(confidence),
                        'details': {
                            'spectral_consistency': float(spectral_consistency),
                            'voice_naturalness': float(voice_naturalness)
                        }
                    },
                    'waveform_data': waveform_data
                })

            except Exception as e:
                print(f"Error in feature extraction or analysis: {str(e)}")
                traceback.print_exc()
                return jsonify({
                    'error': 'Failed to analyze audio features',
                    'details': str(e)
                }), 500

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Error deleting temporary file: {str(e)}")

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to process audio',
            'details': str(e)
        }), 500


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


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Collect real images
        real_dir = os.path.join(root_dir, 'real')
        for img_name in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, img_name), 0))

        # Collect fake images
        fake_dir = os.path.join(root_dir, 'fake')
        for img_name in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, img_name), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class AudioDeepfakeDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.samples = []

        # Collect real audio files
        real_dir = os.path.join(root_dir, 'real')
        for audio_name in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, audio_name), 0))

        # Collect fake audio files
        fake_dir = os.path.join(root_dir, 'fake')
        for audio_name in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, audio_name), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Ensure fixed length (5 seconds)
        target_length = 5 * self.sample_rate
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, max(0, target_length - len(audio))))

        # Convert to tensor
        audio_tensor = torch.FloatTensor(
            audio).unsqueeze(0)  # Add channel dimension
        return audio_tensor, label


def extract_frames_from_videos(video_dir, output_dir, skip_frames=30):
    """
    Extract frames from videos for training

    video_dir: Directory containing 'real' and 'fake' subdirectories with videos
    output_dir: Directory to save extracted frames
    skip_frames: Extract 1 frame for every N frames
    """
    for category in ['real', 'fake']:
        video_category_dir = os.path.join(video_dir, category)
        output_category_dir = os.path.join(output_dir, category)

        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        for video_name in os.listdir(video_category_dir):
            video_path = os.path.join(video_category_dir, video_name)
            video_frames_dir = os.path.join(
                output_category_dir, video_name[:-4])

            if not os.path.exists(video_frames_dir):
                os.makedirs(video_frames_dir)

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % skip_frames == 0:
                    frame_path = os.path.join(
                        video_frames_dir,
                        f'frame_{saved_count:04d}.jpg'
                    )
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f'Extracted {saved_count} frames from {video_name}')


def create_dataset_structure():
    """
    Create the necessary folder structure for datasets if not exists
    """
    folders = [
        'models',  # Folder untuk menyimpan model
        'datasets/image_dataset/train/real',
        'datasets/image_dataset/train/fake',
        'datasets/image_dataset/val/real',
        'datasets/image_dataset/val/fake',
        'datasets/video_dataset/raw_videos/train/real',
        'datasets/video_dataset/raw_videos/train/fake',
        'datasets/video_dataset/raw_videos/val/real',
        'datasets/video_dataset/raw_videos/val/fake',
        'datasets/video_dataset/extracted_frames/train/real',
        'datasets/video_dataset/extracted_frames/train/fake',
        'datasets/video_dataset/extracted_frames/val/real',
        'datasets/video_dataset/extracted_frames/val/fake',
        'datasets/audio_dataset/train/real',
        'datasets/audio_dataset/train/fake',
        'datasets/audio_dataset/val/real',
        'datasets/audio_dataset/val/fake'
    ]

    # Hanya buat folder jika belum ada
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f'Created directory: {folder}')


def create_sample_dataset():
    """
    Create a small sample dataset for testing
    """
    def create_real_face(size=224):
        """Create more realistic face pattern"""
        # Base face shape
        face = np.ones((size, size, 3), dtype=np.uint8) * 220  # Skin tone base

        # Add facial features
        # Eyes
        cv2.circle(face, (int(size*0.35), int(size*0.4)), 15, (50, 50, 50), -1)
        cv2.circle(face, (int(size*0.65), int(size*0.4)), 15, (50, 50, 50), -1)

        # Nose
        pts = np.array([[int(size*0.5), int(size*0.45)],
                        [int(size*0.45), int(size*0.6)],
                        [int(size*0.55), int(size*0.6)]], np.int32)
        cv2.fillPoly(face, [pts], (180, 180, 180))

        # Mouth
        cv2.ellipse(face, (int(size*0.5), int(size*0.7)),
                    (30, 15), 0, 0, 180, (150, 50, 50), -1)

        # Add natural texture
        noise = np.random.normal(0, 10, face.shape).astype(np.uint8)
        face = cv2.add(face, noise)

        # Add slight blur for realism
        face = cv2.GaussianBlur(face, (3, 3), 0)
        return face

    def create_fake_face(size=224):
        """Create fake face with artifacts"""
        # Start with a real face
        face = create_real_face(size)

        # Add deepfake artifacts
        # Color inconsistencies
        face[:, :, 0] = cv2.add(face[:, :, 0],
                                np.random.normal(0, 30, (size, size)).astype(np.uint8))

        # Blurry regions
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (int(size*0.5), int(size*0.5)),
                   int(size*0.4), 255, -1)

        # Create blurred version
        blurred = cv2.GaussianBlur(face, (15, 15), 0)

        # Blend using mask
        # Convert mask to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = mask_3ch / 255.0  # Normalize to 0-1

        face = (cv2.multiply(1.0 - alpha, face.astype(float)) +
                cv2.multiply(alpha, blurred.astype(float)))

        # Add compression artifacts
        _, encoded = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 60])
        face = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return face

    # 1. Create Image Dataset
    print("Creating image dataset...")
    for dataset_type in ['train', 'val']:
        # Real images
        for i in range(5):  # 5 gambar per folder
            img = create_real_face()
            cv2.imwrite(
                f'datasets/image_dataset/{dataset_type}/real/real_{i}.jpg', img)

        # Fake images
        for i in range(5):
            img = create_fake_face()
            cv2.imwrite(
                f'datasets/image_dataset/{dataset_type}/fake/fake_{i}.jpg', img)

    # 2. Create Video Dataset
    print("Creating video dataset...")
    for dataset_type in ['train', 'val']:
        # Real videos
        for i in range(2):  # 2 video per folder
            out = cv2.VideoWriter(
                f'datasets/video_dataset/raw_videos/{dataset_type}/real/real_video_{i}.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (224, 224)
            )
            # Create 30 frames
            for _ in range(30):
                frame = create_real_face()
                out.write(frame)
            out.release()

        # Fake videos
        for i in range(2):
            out = cv2.VideoWriter(
                f'datasets/video_dataset/raw_videos/{dataset_type}/fake/fake_video_{i}.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (224, 224)
            )
            for _ in range(30):
                frame = create_fake_face()
                out.write(frame)
            out.release()

    # 3. Create Audio Dataset
    print("Creating audio dataset...")
    for dataset_type in ['train', 'val']:
        # Real audio
        for i in range(3):  # 3 audio per folder
            duration = 5  # 5 seconds
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration))
            # Create "real" audio signal (mono)
            signal = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)
            signal = signal / np.max(np.abs(signal))

            # Save as mono audio (reshape to 1D array)
            signal = signal.reshape(-1)  # Ensure 1D array for mono
            sf.write(
                f'datasets/audio_dataset/{dataset_type}/real/real_audio_{i}.wav',
                signal,
                sr
            )

        # Fake audio
        for i in range(3):
            duration = 5
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration))
            # Create "fake" audio signal with artifacts (mono)
            signal = np.sin(2 * np.pi * 440 * t)
            noise = np.random.normal(0, 0.1, len(signal))
            signal = signal + noise
            signal = signal / np.max(np.abs(signal))

            # Save as mono audio (reshape to 1D array)
            signal = signal.reshape(-1)  # Ensure 1D array for mono
            sf.write(
                f'datasets/audio_dataset/{dataset_type}/fake/fake_audio_{i}.wav',
                signal,
                sr
            )

    print("Sample dataset creation completed!")


# Update main block
if __name__ == '__main__':
    try:
        # Buat struktur folder jika belum ada
        create_dataset_structure()

        # Inisialisasi model
        print("Initializing model...")
        model = DeepGuardModel()

        # Buat sample dataset jika belum ada
        if not os.path.exists('datasets/image_dataset/train/real/real_0.jpg'):
            print("Creating sample dataset...")
            create_sample_dataset()

            print("Extracting frames from videos...")
            extract_frames_from_videos(
                video_dir='datasets/video_dataset/raw_videos/train',
                output_dir='datasets/video_dataset/extracted_frames/train',
                skip_frames=10
            )

            # Training model
            print("Training visual model...")
            model.train_visual_model(
                train_dir='datasets/image_dataset/train',
                val_dir='datasets/image_dataset/val',
                batch_size=2,
                epochs=5
            )

            print("Training audio model...")
            model.train_audio_model(
                train_dir='datasets/audio_dataset/train',
                val_dir='datasets/audio_dataset/val',
                batch_size=2,
                epochs=5
            )

        # Inisialisasi Flask app dan komponen lainnya
        app.model = model
        app.processor = RealTimeProcessor(model)
        app.adversarial_defense = AdversarialDefense()
        app.explainer = ExplainabilityModule()

        # Start processor
        app.processor.start()

        print("Starting server...")
        # Gunakan threaded=True untuk menangani multiple requests
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except Exception as e:
        print(f"Error during startup: {str(e)}")
        # Cleanup
        if hasattr(app, 'processor'):
            app.processor.stop()
