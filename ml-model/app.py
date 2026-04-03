from flask import Flask, request, jsonify
from model.model import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def get_prediction():

    data = request.get_json()
    strokes = data["strokes"]

    result = predict(strokes)

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(port=5000)
