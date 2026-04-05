"""
game.py - game-state machine. 
"""
import random
import time
from model.model import class_names, predict_topk

#  constants 
MAX_PLAYERS   = 5
NUM_ROUNDS    = 5
ROUND_SECONDS = 20
POINTS_SPEED  = 500   # max speed bonus (full points if correct on first classify)
POINTS_CONF   = 500   # max confidence bonus

# Pool of drawable words — first NUM_CLASSES from the model list, shuffled per game
WORD_POOL = list(class_names)


# helpers

def _score(confidence: float, elapsed: float) -> int:
    """
    confidence : 0-1  (top-k prob for the correct class)
    elapsed    : seconds since round started (0-20)
    """
    conf_pts  = int(confidence * POINTS_CONF)
    speed_pts = int(max(0.0, 1.0 - elapsed / ROUND_SECONDS) * POINTS_SPEED)
    return conf_pts + speed_pts


#  Room 

class Room:
    def __init__(self, room_id: str, host_sid: str, host_name: str):
        self.room_id   = room_id
        self.host_sid  = host_sid

        # sid -> {name, score, scored_this_round}
        self.players: dict[str, dict] = {
            host_sid: {"name": host_name, "score": 0, "scored_this_round": False}
        }

        self.state        = "lobby"   # lobby | countdown | drawing | results | gameover
        self.round_num    = 0         # 1-based when game running
        self.words        = []        # chosen words for this game (NUM_ROUNDS items)
        self.round_start  = 0.0       # time.time() when drawing started
        self.round_strokes: dict[str, list] = {}   # sid → stroke list (latest snapshot)

    # players 

    def add_player(self, sid: str, name: str) -> bool:
        if len(self.players) >= MAX_PLAYERS:
            return False
        if sid not in self.players:
            self.players[sid] = {"name": name, "score": 0, "scored_this_round": False}
        return True

    def remove_player(self, sid: str):
        self.players.pop(sid, None)
        self.round_strokes.pop(sid, None)
        if sid == self.host_sid and self.players:
            self.host_sid = next(iter(self.players))

    def player_list(self):
        return [
            {"sid": sid, "name": p["name"], "score": p["score"],
             "is_host": sid == self.host_sid}
            for sid, p in self.players.items()
        ]

    #  game flow

    def start_game(self) -> bool:
        if self.state != "lobby" or len(self.players) < 1:
            return False
        self.words = random.sample(WORD_POOL, min(NUM_ROUNDS, len(WORD_POOL)))
        for p in self.players.values():
            p["score"] = 0
        self.round_num = 0
        return True

    def begin_round(self):
        self.round_num += 1
        self.state = "drawing"
        self.round_start = time.time()
        self.round_strokes = {sid: [] for sid in self.players}
        for p in self.players.values():
            p["scored_this_round"] = False

    @property
    def current_word(self) -> str:
        if 1 <= self.round_num <= len(self.words):
            return self.words[self.round_num - 1]
        return ""

    # scoring 

    def classify_player(self, sid: str, strokes: list) -> dict:
        """
        Run model on this player's strokes; award points if correct.
        Returns dict with top-k predictions and earned points.
        """
        if not strokes or sid not in self.players:
            return {"topk": [], "earned": 0, "correct": False}

        self.round_strokes[sid] = strokes
        topk = predict_topk(strokes, k=5)
        top_class = topk[0]["class"] if topk else ""
        correct    = (top_class.lower() == self.current_word.lower())
        earned     = 0

        if correct and not self.players[sid]["scored_this_round"]:
            elapsed = time.time() - self.round_start
            conf    = topk[0]["prob"]
            earned  = _score(conf, elapsed)
            self.players[sid]["score"] += earned
            self.players[sid]["scored_this_round"] = True

        return {"topk": topk, "earned": earned, "correct": correct}

    def end_round_scores(self) -> list:
        """Sorted leaderboard snapshot at end of round."""
        return sorted(
            [{"sid": s, "name": p["name"], "score": p["score"]}
             for s, p in self.players.items()],
            key=lambda x: x["score"], reverse=True
        )

    def winner(self) -> dict | None:
        if not self.players:
            return None
        best = max(self.players.items(), key=lambda kv: kv[1]["score"])
        return {"sid": best[0], "name": best[1]["name"], "score": best[1]["score"]}

    def is_last_round(self) -> bool:
        return self.round_num >= NUM_ROUNDS
