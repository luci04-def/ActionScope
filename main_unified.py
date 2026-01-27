import os
import cv2
import torch
import torchvision
import numpy as np
import urllib.request
import shutil
import sys
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from torchvision.models.video import r3d_18, R3D_18_Weights

app = FastAPI(title="ActionAI - Unified")

# Configuration
UPLOADS_DIR = "uploads"
LABELS_PATH = "kinetics_labels.txt"
LABELS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"

os.makedirs(UPLOADS_DIR, exist_ok=True)

# Load labels
if not os.path.exists(LABELS_PATH):
    try:
        urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)
    except Exception as e:
        print(f"Failed to download labels: {e}")

LABELS = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines()]

# Global Model Instance
print("Loading R3D-18 weights...")
try:
    weights = R3D_18_Weights.KINETICS400_V1
    MODEL = r3d_18(weights=weights)
    MODEL.eval()
    print("Model ready.")
except Exception as e:
    print(f"Model load error: {e}")
    MODEL = None

# Preprocessing Constants
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

def preprocess_video(frames):
    processed = []
    for frame in frames:
        frame = cv2.resize(frame, (171, 128))
        h, w, _ = frame.shape
        start_h = (h - 112) // 2
        start_w = (w - 112) // 2
        frame = frame[start_h:start_h+112, start_w:start_w+112]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD
        processed.append(frame)
    
    video_tensor = np.array(processed, dtype=np.float32).transpose(3, 0, 1, 2)
    return torch.from_numpy(video_tensor).unsqueeze(0)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        NUM_FRAMES = 16
        
        if total_frames < NUM_FRAMES:
            cap.release()
            return JSONResponse({"error": "Video too short"}, status_code=400)

        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if len(frames) < NUM_FRAMES:
            return JSONResponse({"error": "Frame extraction failed"}, status_code=500)

        input_batch = preprocess_video(frames)
        with torch.no_grad():
            outputs = MODEL(input_batch)
            probs = torch.softmax(outputs, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()
        
        action = LABELS[pred_idx] if pred_idx < len(LABELS) else f"Class {pred_idx}"
        
        return {
            "action": action.replace("_", " ").title(),
            "confidence": round(confidence * 100, 2),
            "status": "success"
        }
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
async def get_index():
    return HTMLResponse(content=HTML_CONTENT)

# --- EMBEDDED FRONTEND CONTENT (Complete Dashboard with Navigation) ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ActionAI - Neural Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #f8fafc;
            --bg-sidebar: #ffffff;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --border: #e2e8f0;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --sidebar-width: 260px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }

        body {
            background-color: var(--bg-main);
            color: var(--text-dark);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: var(--sidebar-width);
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            padding: 32px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-muted);
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .nav-item.active { background-color: var(--accent-soft); color: var(--accent); }
        .nav-item:hover:not(.active) { background-color: #f1f5f9; }

        /* Main Content */
        .main-content {
            flex: 1;
            padding: 48px;
            overflow-y: auto;
            position: relative;
        }

        .view-section { display: none; }
        .view-section.active { display: block; animation: fadeIn 0.3s ease-out; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

        .header { margin-bottom: 48px; }
        .logo { font-size: 20px; font-weight: 700; color: var(--text-dark); margin-bottom: 40px; }

        /* Analysis View */
        .upload-container {
            max-width: 800px;
            margin: 0 auto;
        }

        .drop-zone {
            width: 100%;
            height: 320px;
            background-color: white;
            border: 2px dashed var(--border);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .drop-zone.dragover { border-color: var(--accent); background-color: var(--accent-soft); }

        .upload-icon {
            width: 64px; height: 64px; background: var(--accent-soft); color: var(--accent);
            border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 24px;
        }

        .upload-btn {
            background-color: var(--accent); color: white; padding: 12px 24px;
            border-radius: 8px; font-weight: 600; border: none; cursor: pointer;
        }

        /* History View */
        .history-list { display: flex; flex-direction: column; gap: 12px; }
        .history-item {
            background: white; border: 1px solid var(--border);
            padding: 16px 20px; border-radius: 12px;
            display: flex; justify-content: space-between; align-items: center;
        }

        /* Settings View */
        .settings-card { background: white; border: 1px solid var(--border); padding: 32px; border-radius: 16px; }
        .setting-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 20px 0; border-bottom: 1px solid #f1f5f9;
        }
        .setting-row:last-child { border-bottom: none; }

        /* Infrastructure Styles */
        .badge { padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 600; }
        .badge-success { background: #dcfce7; color: #15803d; }
        .results-area { margin-top: 40px; }
        .prediction-card { background: white; border: 1px solid var(--border); padding: 24px; border-radius: 12px; display: none; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="logo">NeuralVision AI</div>
        <div class="nav-item active" onclick="switchView('analysis')">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
            Analysis
        </div>
        <div class="nav-item" onclick="switchView('history')">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            History
        </div>
        <div class="nav-item" onclick="switchView('settings')">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
            Settings
        </div>
    </div>

    <div class="main-content">
        <!-- Analysis View -->
        <section id="analysis-view" class="view-section active">
            <div class="header">
                <h1 style="font-size: 24px; font-weight: 700;">Action Recognition</h1>
                <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">Upload a video to analyze spatial-temporal human actions.</p>
            </div>
            <div class="upload-container">
                <div class="drop-zone" id="drop-zone">
                    <input type="file" id="video-input" accept="video/*" hidden>
                    <div class="upload-icon"><svg width="32" height="32" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg></div>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">Drag and drop video here</div>
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 24px;">Support for MP4, MOV, and AVI up to 50MB</div>
                    <button class="upload-btn" onclick="document.getElementById('video-input').click()">Browse files</button>
                    <div id="file-info" style="margin-top:20px; font-size:14px; color:var(--accent); font-weight:500"></div>
                </div>
                <div class="results-area">
                    <div id="loader" style="display:none; text-align:center; color:var(--text-muted)">Analyzing temporal features...</div>
                    <div class="prediction-card" id="pred-card">
                        <div style="font-size:12px; font-weight:700; text-transform:uppercase; color:var(--text-muted); margin-bottom:12px;">Neural Output</div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div id="action-text" style="font-size: 24px; font-weight: 700;">--</div>
                            <div id="conf-badge" class="badge badge-success">0% Conf.</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- History View -->
        <section id="history-view" class="view-section">
            <div class="header">
                <h1 style="font-size: 24px; font-weight: 700;">Audit Trail</h1>
                <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">Review past model inferences and classification records.</p>
            </div>
            <div class="history-list" id="history-list">
                <!-- Dynamic items will appear here -->
                <div class="history-item">
                    <div>
                        <div style="font-weight:600">Sample_Video.mp4</div>
                        <div style="font-size:12px; color:var(--text-muted)">Pre-loaded example</div>
                    </div>
                    <div><span class="badge badge-success">Walking (98%)</span></div>
                </div>
            </div>
        </section>

        <!-- Settings View -->
        <section id="settings-view" class="view-section">
            <div class="header">
                <h1 style="font-size: 24px; font-weight: 700;">Configuration</h1>
                <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">Adjust neural network thresholds and system parameters.</p>
            </div>
            <div class="settings-card">
                <div class="setting-row">
                    <div>
                        <div style="font-weight:600">Confidence Threshold</div>
                        <div style="font-size:13px; color:var(--text-muted)">Ignore results below this percentage</div>
                    </div>
                    <input type="range" min="0" max="100" value="85" style="width: 150px;">
                </div>
                <div class="setting-row">
                    <div>
                        <div style="font-weight:600">High Resolution Processing</div>
                        <div style="font-size:13px; color:var(--text-muted)">Increase accuracy (slower processing)</div>
                    </div>
                    <input type="checkbox" checked>
                </div>
                <div class="setting-row">
                    <div>
                        <div style="font-weight:600">Auto-Save History</div>
                        <div style="font-size:13px; color:var(--text-muted)">Store all analysis results locally</div>
                    </div>
                    <input type="checkbox" checked>
                </div>
            </div>
        </section>
    </div>

    <script>
        function switchView(viewId) {
            document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById(viewId + '-view').classList.add('active');
            event.currentTarget.classList.add('active');
        }

        const dropZone = document.getElementById('drop-zone');
        const videoInput = document.getElementById('video-input');
        const fileInfo = document.getElementById('file-info');
        const loader = document.getElementById('loader');
        const predCard = document.getElementById('pred-card');
        const actionText = document.getElementById('action-text');
        const confBadge = document.getElementById('conf-badge');
        const historyList = document.getElementById('history-list');

        dropZone.onclick = (e) => { if(e.target.tagName !== 'BUTTON') videoInput.click(); };
        videoInput.onchange = (e) => handleFiles(e.target.files[0]);
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        dropZone.ondrop = (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFiles(e.dataTransfer.files[0]); };

        async function handleFiles(file) {
            if (!file || !file.type.startsWith('video/')) return;
            const currentName = file.name;
            fileInfo.textContent = `Processing: ${currentName}`;
            loader.style.display = 'block';
            predCard.style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                loader.style.display = 'none';
                predCard.style.display = 'block';
                actionText.textContent = data.action;
                confBadge.textContent = `${data.confidence}% Conf.`;
                
                // ADD TO HISTORY
                addToHistory(currentName, data.action, data.confidence);
            } catch (err) {
                alert('Analysis failed');
                loader.style.display = 'none';
            }
        }

        function addToHistory(filename, action, confidence) {
            const now = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div>
                    <div style="font-weight:600">${filename}</div>
                    <div style="font-size:12px; color:var(--text-muted)">Just now, ${now}</div>
                </div>
                <div><span class="badge badge-success">${action} (${confidence}%)</span></div>
            `;
            // Insert at the top of the history list
            historyList.insertBefore(item, historyList.firstChild);
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
