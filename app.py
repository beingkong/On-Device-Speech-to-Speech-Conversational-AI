from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/')
def index():
    return "Welcome to the Speech-to-Speech API!"

@app.route('/stt', methods=['POST'])
def speech_to_text():
    # This is where the speech-to-text logic will go
    # For now, just return a dummy response
    audio_data = request.data # Assuming audio data is sent directly in the request body
    print(f"Received audio data of length: {len(audio_data)} bytes")
    return jsonify({"text": "This is a dummy transcription."})

@app.route('/llm', methods=['POST'])
def llm_response():
    data = request.get_json()
    text = data.get('text', '')
    print(f"Received text for LLM: {text}")
    return jsonify({"response": f"LLM received: {text}"})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    # This is where the text-to-speech logic will go
    # For now, just return a dummy response
    data = request.get_json()
    text = data.get('text', '')
    print(f"Received text for TTS: {text}")
    # In a real scenario, you would synthesize audio and return it
    # For now, we'll just return a placeholder
    return jsonify({"audio_url": "/dummy_audio.mp3"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)