from flask import Flask, request, jsonify
import torch
import numpy as np

from model.model import DoodleModel

import os

class_files = [
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

class_names = [c.replace(".csv","") for c in class_files]

app = Flask(__name__)

model = DoodleModel()
model.load_state_dict(torch.load("model/rnn_model.pth", map_location="cpu"))
model.eval()


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

    seq.append([0,0,0,0,1])

    return seq


@app.route("/predict", methods=["POST"])
def predict():

    strokes = request.json["strokes"]

    seq = strokes_to_5d(strokes)

    seq = np.array(seq, dtype=np.float32)

    length = len(seq)

    x = torch.tensor(seq).unsqueeze(0)
    lengths = torch.tensor([length])

    with torch.no_grad():

        output,_ = model(x,lengths)

        pred = torch.argmax(output,dim=1).item()

        label = class_names[pred]

        return jsonify({"prediction": label})


if __name__ == "__main__":
    app.run(port=5000)