from flask import Flask

app = Flask(__name__)


@app.route("/")
def predict_speech_command():
    return "Hello!"


if __name__ == "__main__":
    app.run(debug=False)
