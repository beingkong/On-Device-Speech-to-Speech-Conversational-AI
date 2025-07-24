
import sys
import os
import threading
from flask import Flask, render_template
from flask_sock import Sock
from src.services.model_server import ModelServer
from src.services.conversation_manager import ConversationManager

app = Flask(__name__, template_folder='src/web/templates', static_folder='src/web/static')
sock = Sock(app)

# Load models once at startup
print("Initializing ModelServer...")
model_server = ModelServer()
print("ModelServer initialized.")

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@sock.route('/audio_stream')
def ws(ws_client):
    """Handle WebSocket connections."""
    print("WebSocket connection established.")
    
    # Create a new conversation manager for each client
    conversation_manager = ConversationManager(
        vad_model=model_server.vad_model,
        stt_processor=model_server.stt_processor,
        stt_model=model_server.stt_model,
        tts_engine=model_server.tts_engine
    )
    
    # Start threads for handling user input and AI output
    user_input_thread = threading.Thread(target=conversation_manager.handle_user_input, args=(ws_client,))
    ai_output_thread = threading.Thread(target=conversation_manager.handle_ai_output, args=(ws_client,))
    
    user_input_thread.start()
    ai_output_thread.start()
    
    # Wait for threads to complete
    user_input_thread.join()
    ai_output_thread.join()
    
    print("WebSocket connection closed.")

if __name__ == "__main__":
    print("Starting Flask server...")
    # Note: Using 'gevent' as the WebSocket server is common, but for simplicity
    # and since flask-sock supports it, the default Werkzeug server is used.
    # For production, consider a more robust server like gevent or gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=False)
