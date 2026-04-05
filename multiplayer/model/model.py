import torch
import torch.nn as nn

# ---------------- CONFIG ---------------- #

MAX_SEQ_LEN = 200

all_files = [
'The Eiffel Tower.csv','The Great Wall of China.csv','The Mona Lisa.csv','airplane.csv',
'alarm clock.csv','ambulance.csv','angel.csv','animal migration.csv','ant.csv','anvil.csv',
'apple.csv','arm.csv','asparagus.csv','axe.csv','backpack.csv','banana.csv','bandage.csv',
'barn.csv','baseball bat.csv','baseball.csv','basket.csv','basketball.csv','bat.csv',
'bathtub.csv','beach.csv','bear.csv','beard.csv','bed.csv','bee.csv','belt.csv','bench.csv',
'bicycle.csv','binoculars.csv','bird.csv','birthday cake.csv','blackberry.csv','blueberry.csv',
'book.csv','boomerang.csv','bottlecap.csv','bowtie.csv','bracelet.csv','brain.csv','bread.csv',
'bridge.csv','broccoli.csv','broom.csv','bucket.csv','bulldozer.csv','bus.csv','bush.csv',
'butterfly.csv','cactus.csv','cake.csv','calculator.csv','calendar.csv','camel.csv','camera.csv',
'camouflage.csv','campfire.csv','candle.csv','cannon.csv','canoe.csv','car.csv','carrot.csv',
'castle.csv','cat.csv','ceiling fan.csv','cell phone.csv','cello.csv','chair.csv','chandelier.csv',
'church.csv','circle.csv','clarinet.csv','clock.csv','cloud.csv','coffee cup.csv','compass.csv',
'computer.csv','cookie.csv','cooler.csv','couch.csv','cow.csv','crab.csv','crayon.csv',
'crocodile.csv','crown.csv','cruise ship.csv','cup.csv','diamond.csv','dishwasher.csv',
'diving board.csv','dog.csv','dolphin.csv','donut.csv','door.csv','dragon.csv','dresser.csv',
'drill.csv','drums.csv','duck.csv','dumbbell.csv','ear.csv','elbow.csv','elephant.csv',
'envelope.csv','eraser.csv','eye.csv','eyeglasses.csv','face.csv','fan.csv','feather.csv',
'fence.csv','finger.csv','fire hydrant.csv','fireplace.csv','firetruck.csv','fish.csv',
'flamingo.csv','flashlight.csv','flip flops.csv','floor lamp.csv','flower.csv','flying saucer.csv',
'foot.csv','fork.csv','frog.csv','frying pan.csv','garden hose.csv','garden.csv','giraffe.csv',
'goatee.csv','golf club.csv','grapes.csv','grass.csv','guitar.csv','hamburger.csv','hammer.csv',
'hand.csv','harp.csv','hat.csv','headphones.csv','hedgehog.csv','helicopter.csv','helmet.csv',
'hexagon.csv','hockey puck.csv','hockey stick.csv','horse.csv','hospital.csv','hot air balloon.csv',
'hot dog.csv','hot tub.csv','hourglass.csv','house plant.csv','house.csv','hurricane.csv',
'ice cream.csv','jacket.csv','jail.csv','kangaroo.csv','key.csv','keyboard.csv','knee.csv',
'ladder.csv','lantern.csv','laptop.csv','leaf.csv','leg.csv','light bulb.csv','lighthouse.csv',
'lightning.csv','line.csv','lion.csv','lipstick.csv','lobster.csv','lollipop.csv','mailbox.csv',
'map.csv','marker.csv','matches.csv','megaphone.csv','mermaid.csv','microphone.csv',
'microwave.csv','monkey.csv','moon.csv','mosquito.csv','motorbike.csv','mountain.csv',
'mouse.csv','moustache.csv','mouth.csv','mug.csv','mushroom.csv','nail.csv','necklace.csv',
'nose.csv','ocean.csv'
]

NUM_CLASSES = 100
class_files = all_files[:NUM_CLASSES]
class_names = [c.replace(".csv","") for c in class_files]


# ---------------- PREPROCESS ---------------- #

def strokes_to_5d(strokes):
    seq = []
    
    for stroke in strokes:
        x, y = stroke
        
        for i in range(len(x)):
            if i == 0:
                dx, dy = 0, 0
            else:
                dx = (x[i] - x[i-1]) / 255.0
                dy = (y[i] - y[i-1]) / 255.0
            
            seq.append([dx, dy, 1, 0, 0])
        
        seq[-1][2] = 0
        seq[-1][3] = 1
    
    seq.append([0, 0, 0, 0, 1])
    return seq


def pad_sequence(seq, max_len=MAX_SEQ_LEN):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [[0,0,0,0,0]] * (max_len - len(seq))


def preprocess(strokes):
    seq = strokes_to_5d(strokes)
    length = min(len(seq), MAX_SEQ_LEN)
    seq = pad_sequence(seq, MAX_SEQ_LEN)

    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    length = torch.tensor([length])

    return seq, length

# ---------------- MODEL ---------------- #

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out, lengths):
        B, T, _ = lstm_out.size()

        scores = torch.tanh(self.attn(lstm_out))
        scores = self.context(scores).squeeze(-1)

        mask = torch.arange(T).expand(B, T) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e4)

        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)

        return context, weights


class DoodleModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, num_classes=NUM_CLASSES):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.attn = Attention(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )

    def forward(self,x,lengths,hidden=None):
        lstm_out,_ = self.lstm(x,hidden)

        idx = (lengths-1).unsqueeze(1).unsqueeze(2).expand(-1,1,lstm_out.size(2))
        last_hidden = lstm_out.gather(1,idx).squeeze(1)

        context,_ = self.attn(lstm_out,lengths)

        combined = context + last_hidden
        out = self.fc(combined)

        return out, None

# ---------------- LOAD MODEL ---------------- #

model = DoodleModel()
model.load_state_dict(torch.load("model/ig.pth", map_location="cpu"))
model.eval()

# ---------------- PREDICT ---------------- #

def predict(strokes):
    print("RAW STROKES:", strokes[:1])  # 👈 check structure

    seq, length = preprocess(strokes)

    print("SEQ SAMPLE:", seq[0][:5])   # 👈 check values

    with torch.no_grad():
        output,_ = model(seq, length)
        pred = torch.argmax(output, dim=1).item()

    return class_names[pred]


def predict_topk(strokes, k=3):
    seq, length = preprocess(strokes)

    with torch.no_grad():
        output,_ = model(seq, length)
        probs = torch.softmax(output, dim=1)
        topk = torch.topk(probs, k)

    return [
        {"class": class_names[i], "prob": float(p)}
        for i, p in zip(topk.indices[0], topk.values[0])
    ]
