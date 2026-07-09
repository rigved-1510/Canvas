# DrawGuessAI

A real-time AI-powered sketch recognition and multiplayer guessing game. Draw anything — the model predicts what you're drawing **live**, stroke by stroke. Challenge friends in a room-based multiplayer mode where your score depends on both **how accurately** you draw and **how fast** the AI recognises it.

**Department of Information Technology**  
*National Institute of Technology Karnataka (NITK) - Surathkal, India*

---

# Features

-  **Real-time Sketch Recognition** — Live predictions update as you draw, using a 2-layer LSTM with a custom attention mechanism trained on 200 sketch classes from the Google Quick, Draw! dataset.
-  **Sequence-based Deep Learning** — Strokes are encoded as 5D temporal sequences (Δx, Δy, pen-down, pen-up, end-of-sketch) and processed by the model to capture drawing order and style.
-  **Stroke Preprocessing Pipeline** — Client-side downsampling, Gaussian smoothing, and coordinate normalisation (scaled to 255) ensure clean, consistent input before inference.
-  **Multiplayer Game Engine** — Room-based multiplayer (up to 5 players) with 5 rounds of 20 seconds each, real-time leaderboard, and a confidence + speed scoring formula.
-  **LAN / Hotspot Play** — No internet required; players connect over the same Wi-Fi or mobile hotspot using the host's network IP.
-  **Top-K Predictions** — The model returns the top-5 most likely classes with confidence scores for each player during live gameplay.

---

# Tech Stack

| Layer | Technology |
| :--- | :--- |
| ML Model | Python, PyTorch (LSTM + Attention) |
| Model Training | Jupyter Notebook (`rnn_classifier.ipynb`) |
| ML API Server | Python, Flask |
| Solo Backend | Node.js, Express, Socket.IO, Axios |
| Multiplayer Server | Python, Flask, Flask-SocketIO |
| Frontend (Solo) | HTML5 Canvas, JavaScript, Socket.IO Client |
| Frontend (Multiplayer) | HTML5 Canvas, Jinja2 Templates, Socket.IO Client |

---

# Project Structure

```text
DrawGuessAI/
│
├── frontend/               # Solo mode UI
│   ├── index.html
│   └── script.js           # Canvas drawing, stroke preprocessing, Socket.IO client
│
├── backend/                # Solo mode server (Node.js)
│   ├── server.js           # Socket.IO server, proxies strokes to Flask ML API
│   └── package.json
│
├── ml-model/               # Standalone ML inference server
│   ├── app.py              # Flask REST API — POST /predict
│   ├── requirements.txt
│   └── model/
│       ├── model.py        # DoodleModel (LSTM + Attention), preprocessing, prediction
│       └── rnn_model.pth   # Trained model weights (200 classes)
│
│   └── notebook/
│       └── rnn_classifier.ipynb   # Training notebook
│
└── multiplayer/            # Self-contained multiplayer game
    ├── app.py              # Flask + SocketIO server — room management, game loop
    ├── game.py             # Room state machine, scoring logic
    ├── requirements.txt
    └── model/
        ├── model.py        # DoodleModel (100 classes), top-k prediction
        └── ig.pth          # Trained model weights
    └── templates/
        └── index.html      # Full multiplayer game UI
```

---

# How to Run Locally

## Prerequisites

- Python 3.10+
- Node.js 18+
- `pip`
- `npm`

---

## Mode 1 — Solo (Real-time Prediction)

This mode uses three separate processes: the ML Flask server, the Node.js backend, and the frontend.

### Step 1 — Start the ML Inference Server

```bash
cd ml-model
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

### Step 2 — Start the Node.js Backend

```bash
cd backend
npm install
node server.js
# Runs on http://localhost:3000
```

### Step 3 — Open the Frontend

Open `frontend/index.html` directly in your browser (or serve it using any static server).

> Draw on the canvas — predictions appear live after each stroke.

---

## Mode 2 — Multiplayer (Room-based Game)

This mode is fully self-contained inside the `multiplayer/` directory.

### Step 1 — Install Dependencies

```bash
cd multiplayer
pip install -r requirements.txt
```

### Step 2 — Run the Server

```bash
python app.py
```

You will see:

```text
════════════════════════════════════════════════════════════
  Quick Draw Multiplayer
════════════════════════════════════════════════════════════
  Local:    http://127.0.0.1:5000
  Network:  http://<your-local-ip>:5000
  Share the Network link with players on the same Wi-Fi / hotspot
════════════════════════════════════════════════════════════
```

### Step 3 — Players Join

- **Host** opens `http://127.0.0.1:5000`, creates a room, and shares the 4-letter room code.
- **Other players** open the **Network URL** on any device connected to the same Wi-Fi or hotspot and join using the room code.
- The host presses **Start Game** when everyone is in the lobby.

---

## Multiplayer Scoring

| Bonus | Max Points | Criteria |
| :--- | :---: | :--- |
| Confidence Bonus | 500 | Model's confidence that your drawing matches the word |
| Speed Bonus | 500 | How early in the 20-second window the AI recognised it |
| **Total per round** | **1000** | |

---

# Model Architecture

```text
Input: 5D stroke sequence
[Δx, Δy, pen-down, pen-up, end] (max 200 steps)
         ↓
2-Layer LSTM (hidden size: 256)
         ↓
Custom Attention Layer (context vector over all time steps)
         ↓
Fully Connected Head
(256 → 128 → ReLU → Dropout 0.3 → num_classes)
         ↓
Output:
Class logits
(200 classes — Solo / 100 classes — Multiplayer)
```

Trained on the **Google Quick, Draw! Dataset**.
