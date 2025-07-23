import threading
import queue
import time
import numpy as np
import torch
import json
from config.settings import settings
from components.llm.llm_client import get_ai_response, parse_stream_chunk
 # ChatMLSample, Message 已移除，直接用 dict 结构传递文本
from transformers import AutoProcessor, VoxtralForConditionalGeneration
import soundfile as sf
import tempfile

class ConversationManager:
    def __init__(self, vad_model, stt_processor, stt_model, tts_engine):
        self.llm_text_queue = queue.Queue()
        self.ai_is_speaking = threading.Event()
        self.interruption_event = threading.Event()
        self.messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Receive pre-loaded models
        self.vad_model = vad_model
        self.stt_processor = stt_processor
        self.stt_model = stt_model
        self.tts_engine = tts_engine

    def handle_user_input(self, ws):
        audio_buffer = []
        speech_started = False
        silence_start_time = None
        
        vad_chunk_size = 512 
        internal_buffer = np.array([], dtype=np.int16)

        while ws.connected:
            try:
                data = ws.receive(timeout=0.1)
                if data and isinstance(data, (bytes, bytearray)):
                    
                    new_samples = np.frombuffer(data, dtype=np.int16)
                    internal_buffer = np.concatenate([internal_buffer, new_samples])

                    while len(internal_buffer) >= vad_chunk_size:
                        chunk_to_process = internal_buffer[:vad_chunk_size]
                        internal_buffer = internal_buffer[vad_chunk_size:]

                        is_speech = self.vad_model(torch.from_numpy(chunk_to_process).to(self.device), 16000).item() > 0.5

                        if is_speech:
                            if not speech_started: print("Speech started.")
                            speech_started = True
                            silence_start_time = None
                            audio_buffer.append(chunk_to_process)
                        elif speech_started:
                            audio_buffer.append(chunk_to_process)
                            if silence_start_time is None: silence_start_time = time.time()
                            
                            if time.time() - silence_start_time > settings.VAD_SILENCE_THRESHOLD:
                                print("Speech ended.")
                                speech_started = False
                                silence_start_time = None
                                
                                if audio_buffer:
                                    full_audio_np = np.concatenate(audio_buffer)
                                    audio_buffer = []
                                    
                                    ws.send(json.dumps({"status": "transcribing"}))
                                    
                                    # Save audio to a temporary file to ensure processor compatibility.
                                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                                        # Write the int16 numpy array directly to a WAV file.
                                        sf.write(tmp_file.name, full_audio_np, 16000)

                                        # 1. Create the conversation input using the file path.
                                        conversation = [
                                            {"role": "user", "content": [
                                                {"type": "audio", "path": tmp_file.name}
                                            ]}
                                        ]
                                    
                                        # 2. Apply the template to get model-ready inputs.
                                        inputs = self.stt_processor.apply_chat_template(
                                            conversation
                                        ).to(self.device, dtype=torch.bfloat16)

                                        # 3. Generate the token IDs.
                                        predicted_ids = self.stt_model.generate(**inputs)
                                        
                                        # 4. Decode the result.
                                        transcribed_text = self.stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                                    
                                    if transcribed_text and transcribed_text.strip():
                                        print(f"User: {transcribed_text}")
                                        self.messages.append({"role": "user", "content": transcribed_text})
                                        
                                        llm_thread = threading.Thread(target=self.get_llm_response)
                                        llm_thread.start()
                                    
                                    if transcribed_text and transcribed_text.strip():
                                        print(f"User: {transcribed_text}")
                                        self.messages.append({"role": "user", "content": transcribed_text})
                                        
                                        llm_thread = threading.Thread(target=self.get_llm_response)
                                        llm_thread.start()

            except Exception as e:
                if not isinstance(e, TimeoutError):
                    print(f"Error in user input thread: {e}")
                continue


    def get_llm_response(self):
        import requests
        session = requests.Session()
        
        try:
            response_stream = get_ai_response(
                session=session,
                messages=self.messages,
                llm_model=settings.LLM_MODEL,
                llm_url=settings.OLLAMA_URL,
                max_tokens=settings.MAX_TOKENS,
                stream=True
            )
            
            full_response = ""
            for chunk in response_stream:
                if self.interruption_event.is_set(): break
                data = parse_stream_chunk(chunk)
                if data and "choices" in data and "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                    content = data["choices"][0]["delta"]["content"]
                    if content:
                        full_response += content
                        self.llm_text_queue.put(content)
            
            if full_response: self.messages.append({"role": "assistant", "content": full_response})
            self.llm_text_queue.put(None)

        except Exception as e:
            print(f"Error getting LLM response: {e}")
            self.llm_text_queue.put(None)

    def handle_ai_output(self, ws):
        sentence_buffer = ""
        while ws.connected:
            try:
                text_chunk = self.llm_text_queue.get(timeout=0.1)
                if text_chunk is None:  # End of stream signal
                    if sentence_buffer:
                        # If there's a leftover sentence, process it
                        self._generate_and_send_audio(sentence_buffer, ws)
                        sentence_buffer = ""
                    self.ai_is_speaking.clear()
                    self.interruption_event.clear()
                    continue

                sentence_buffer += text_chunk

                # Process buffer if it contains sentence-ending punctuation
                if any(p in sentence_buffer for p in ".!?"):
                    # Split buffer into sentences
                    sentences = re.split(r'(?<=[.!?])\s*', sentence_buffer)
                    # The last part might be an incomplete sentence, so keep it in the buffer
                    sentence_buffer = sentences.pop() if sentences[-1] and not sentences[-1].endswith(('.','!','?')) else ""
                    
                    for sentence in sentences:
                        if sentence.strip():
                            self._generate_and_send_audio(sentence.strip(), ws)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in AI output thread: {e}")
                self.ai_is_speaking.clear()
                self.interruption_event.clear()

    def _generate_and_send_audio(self, text, ws):
        if self.interruption_event.is_set():
            # Clear the queue if interrupted
            while not self.llm_text_queue.empty(): self.llm_text_queue.get()
            self.ai_is_speaking.clear()
            self.interruption_event.clear()
            return
            
        if not self.ai_is_speaking.is_set(): self.ai_is_speaking.set()

        print(f"Generating audio for: '{text}'")

        # Prepare input for Higgs-Audio
        # 直接用 dict 结构传递文本给 TTS
        chat_ml_sample = {"role": "assistant", "content": text}

        # Generate audio
        audio_output = self.tts_engine.generate(chat_ml_sample)
        
        # Higgs-Audio output is a dict, we need the audio bytes
        # Assuming the audio is in 'audio' key and is a numpy array
        if audio_output and 'audio' in audio_output and audio_output['audio'] is not None:
             # Convert numpy array to bytes
            audio_bytes = audio_output['audio'].tobytes()
            ws.send(audio_bytes)
        else:
            print(f"Warning: Higgs-Audio did not return audio for text: '{text}'")
