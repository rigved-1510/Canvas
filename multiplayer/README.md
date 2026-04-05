# Draw - Guess


## Project Structure

```
multiplayer/
├── app.py              Flask + SocketIO server (run this)
├── game.py             Game state machine
├── requirements.txt
├── model/
│   ├── __init__.py
│   ├── model.py        (unchanged logic)
│   └── ig.pth           PUT YOUR .pth FILE HERE
└── templates/
    └── index.html       Full game UI
```

---

## Setup


### 1. Install dependencies

```bash
cd quickdraw
pip install -r requirements.txt
```

> If you use a virtualenv (recommended):
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### 3. Run the server

```bash
python app.py
```

You'll see something like:

```
════════════════════════════════════════════════════
    Quick Draw Multiplayer
════════════════════════════════════════════════════
  Local:    http://127.0.0.1:5000
  Network:  http://192.168.1.42:5000
  Share the Network link with players on the same Wi-Fi / hotspot
════════════════════════════════════════════════════
```

### 4. Players join

- **Host** opens `http://127.0.0.1:5000` (or the network URL)
- **Other players** open the **Network URL** on any device connected to the same Wi-Fi / hotspot
- Players create/join a room using the 4-letter room code

---

## Game Rules

| Setting         | Value          |
|-----------------|----------------|
| Players per room | 1–5           |
| Rounds          | 5              |
| Time per round  | 20 seconds     |
| Max points/round | 1000          |

**Scoring formula:**
- **Confidence bonus** (0–500): based on model's confidence that your drawing is correct
- **Speed bonus** (0–500): based on how early in the 20s window the AI recognised it

---

## Changing the number of classes

Edit **one line** in `model/model.py`:

```python
NUM_CLASSES = 100   # change to 150, 200, etc.
```

Make sure your `.pth` file was trained with the same number of classes.

The word pool is always the first `NUM_CLASSES` entries from `_ALL_CLASS_FILES`.

---

## Hotspot Setup (no router needed)

1. On the host machine, enable a mobile hotspot (or use your phone as a hotspot)
2. Connect all player devices to that hotspot
3. Run `python app.py` — use the **Network IP** shown in the terminal
4. Share that URL with all players

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Room not found" | Check the 4-letter code; codes are case-insensitive |
| Players can't connect | Make sure everyone is on the **same** Wi-Fi / hotspot |
| Firewall blocking | Allow port 5000 in your OS firewall settings |
| Model not loading | Confirm `model/ig.pth` exists and matches `NUM_CLASSES` |
| Slow classification | Normal on CPU; inference runs per player every 1.5s |
