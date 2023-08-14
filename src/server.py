from flask import Flask, request, jsonify
from commands_spotting_service import CommandsSpottingService

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict_speech_command():
    audio = request.files["file"]
    service = CommandsSpottingService(model_path="model.h5")
    return service.predict(audio)


if __name__ == "__main__":
    app.run(debug=False)
