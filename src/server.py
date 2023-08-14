from flask import Flask, request, jsonify
from commands_spotting_service import CommandsSpottingService
import random
import os

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict_speech_command():
    try:
        service = CommandsSpottingService(model_path="model.h5")
        audio = request.files["file"]
        return service.predict(audio)
    except:
        return "None"


if __name__ == "__main__":
    app.run(debug=True)
