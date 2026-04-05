"""
app.py  multiplayer server
Run:  python app.py
"""
import random
import string
import threading
import time

from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room, emit

from game import Room, NUM_ROUNDS, ROUND_SECONDS

# app setup
app = Flask(__name__)
app.config["SECRET_KEY"] = "quickdraw-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

rooms: dict[str, Room] = {}   # room_id 
sid_to_room: dict[str, str] = {}   # sid 

#  helpers 
def _make_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=4))


def _get_room(sid: str) -> Room | None:
    rid = sid_to_room.get(sid)
    return rooms.get(rid) if rid else None


def _broadcast_lobby(room: Room):
    socketio.emit("lobby_update", {
        "players": room.player_list(),
        "room_id": room.room_id,
        "host_sid": room.host_sid,
    }, room=room.room_id)


def _run_round(room: Room):
    """Called in a background thread for each round."""
    word = room.current_word
    socketio.emit("round_start", {
        "round": room.round_num,
        "total_rounds": NUM_ROUNDS,
        "word": word,
        "seconds": ROUND_SECONDS,
    }, room=room.room_id)

    # tick every second so clients can show live countdown
    for remaining in range(ROUND_SECONDS - 1, -1, -1):
        time.sleep(1)
        if room.state != "drawing":
            return   # aborted (e.g. everyone left)
        socketio.emit("tick", {"remaining": remaining}, room=room.room_id)

    #  collect final classifications
    room.state = "results"
    final_results = []
    for sid in list(room.players):
        strokes = room.round_strokes.get(sid, [])
        res = room.classify_player(sid, strokes)
        final_results.append({
            "sid": sid,
            "name": room.players[sid]["name"],
            "correct": res["correct"],
            "earned": res["earned"],
            "topk": res["topk"],
        })

    leaderboard = room.end_round_scores()
    socketio.emit("round_end", {
        "round": room.round_num,
        "word": word,
        "results": final_results,
        "leaderboard": leaderboard,
        "is_last": room.is_last_round(),
    }, room=room.room_id)

    if room.is_last_round():
        time.sleep(5)
        room.state = "gameover"
        socketio.emit("game_over", {
            "winner": room.winner(),
            "leaderboard": leaderboard,
        }, room=room.room_id)
    else:
        time.sleep(5)
        _advance_round(room)


def _advance_round(room: Room):
    room.begin_round()
    threading.Thread(target=_run_round, args=(room,), daemon=True).start()


# HTTP route 

@app.route("/")
def index():
    return render_template("index.html")


# Socket events 

@socketio.on("connect")
def on_connect():
    pass   # nothing needed


@socketio.on("disconnect")
def on_disconnect():
    room = _get_room(request.sid)
    if not room:
        return
    room.remove_player(request.sid)
    sid_to_room.pop(request.sid, None)

    if not room.players:
        rooms.pop(room.room_id, None)
        return

    _broadcast_lobby(room)
    if room.state in ("drawing", "results"):
        # notify remaining players someone left mid-game
        emit("player_left", {"name": "A player"}, room=room.room_id)


@socketio.on("create_room")
def on_create_room(data):
    name = (data.get("name") or "Host").strip()[:16]
    rid  = _make_room_id()
    while rid in rooms:
        rid = _make_room_id()

    room = Room(rid, request.sid, name)
    rooms[rid] = room
    sid_to_room[request.sid] = rid
    join_room(rid)

    emit("room_created", {"room_id": rid, "sid": request.sid})
    _broadcast_lobby(room)


@socketio.on("join_room_req")
def on_join_room(data):
    rid  = (data.get("room_id") or "").strip().upper()
    name = (data.get("name") or "Player").strip()[:16]

    if rid not in rooms:
        emit("error", {"msg": "Room not found. Check the code and try again."})
        return

    room = rooms[rid]
    if room.state != "lobby":
        emit("error", {"msg": "Game already in progress."})
        return
    if not room.add_player(request.sid, name):
        emit("error", {"msg": "Room is full (max 5 players)."})
        return

    sid_to_room[request.sid] = rid
    join_room(rid)

    emit("room_joined", {"room_id": rid, "sid": request.sid})
    _broadcast_lobby(room)


@socketio.on("start_game")
def on_start_game():
    room = _get_room(request.sid)
    if not room or request.sid != room.host_sid:
        emit("error", {"msg": "Only the host can start the game."})
        return
    if not room.start_game():
        emit("error", {"msg": "Cannot start game right now."})
        return

    socketio.emit("game_starting", {"rounds": NUM_ROUNDS}, room=room.room_id)
    time.sleep(1)
    _advance_round(room)


@socketio.on("submit_strokes")
def on_submit_strokes(data):
    """
    Client sends its stroke data every few seconds for live classification
    AND once more at round end.  We classify immediately and emit back
    only to that player so they get live feedback.
    """
    room = _get_room(request.sid)
    if not room or room.state != "drawing":
        return

    strokes = data.get("strokes", [])
    if not strokes:
        return

    res = room.classify_player(request.sid, strokes)

    # send live feedback only to this player
    emit("live_result", {
        "topk": res["topk"],
        "correct": res["correct"],
        "earned": res["earned"],
        "word": room.current_word,
    })

    if res["correct"] and res["earned"] > 0:
        # broadcast to room that someone got it (without revealing the word to spectators)
        socketio.emit("player_scored", {
            "name": room.players[request.sid]["name"],
        }, room=room.room_id)


@socketio.on("play_again")
def on_play_again():
    room = _get_room(request.sid)
    if not room or request.sid != room.host_sid:
        return
    room.state = "lobby"
    _broadcast_lobby(room)


# run 
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "═" * 52)
    print("  Quick Draw Multiplayer")
    print("═" * 52)
    print(f"  Local:    http://127.0.0.1:5000")
    print(f"  Network:  http://{local_ip}:5000")
    print("  Share the Network link with players on the same Wi-Fi / hotspot")
    print("═" * 52 + "\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
