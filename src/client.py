import requests

URL = "http://127.0.0.1:5000"
AUDIO_PATH = "audios/right/0d393936_nohash_0.wav"

file = open(AUDIO_PATH, "rb")
values = {"file": (AUDIO_PATH, file, "audio/wav")}
response = requests.post(URL, files=values)
data = response.text

print(f"Predicted Command is {data}")
