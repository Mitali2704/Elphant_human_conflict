from flask import Flask, request, jsonify
from flask_cors import CORS  # allow cross-origin requests (important for frontend)

app = Flask(__name__)
CORS(app)  # allow all origins

events = []  # in-memory storage of detection events

@app.route("/upload_event", methods=["POST"])
def upload_event():
    data = request.json
    events.append(data)
    return jsonify({"status": "success", "event": data}), 200

@app.route("/get_events", methods=["GET"])
def get_events():
    return jsonify(events)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
