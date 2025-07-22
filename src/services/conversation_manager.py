import threading
import queue
import time
import numpy as np
import torch
import json
from config.settings import settings
from components.llm.llm_client import get_ai_response, parse_stream_chunk
from components.voice.voice_manager import VoiceGenerator
from transformers import AutoProcessor, VoxtralForConditionalGeneration

class ConversationManager:
    def __init__(self, vad_model, stt_processor, stt_model, voice_generator):
        self.llm_text_queue = queue.Queue()
        self.ai_is_speaking = threading.Event()
        self.interruption_event = threading.Event()
        self.messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Receive pre-loaded models
        self.vad_model = vad_model
        self.stt_processor = stt_processor
        self.stt_model = stt_model
        self.voice_generator = voice_generator

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
                                    
                                    # Manual STT processing, step-by-step
                                    audio_float32 = full_audio_np.astype(np.float32) / 32768.0
                                    
                                    # 1. Feature Extraction
                                    input_features = self.stt_processor.feature_extractor(
                                        [audio_float32], 
                                        sampling_rate=16000, 
                                        return_tensors="pt"
                                    ).input_features.to(self.device, dtype=torch.bfloat16)

                                    # 2. Model Inference - Direct model call for speech recognition
                                    predicted_ids = self.stt_model(input_features).logits
                                    
                                    # 3. Decoding
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
        while ws.connected:
            try:
                text_chunk = self.llm_text_queue.get(timeout=0.1)
                if text_chunk is None:
                    self.ai_is_speaking.clear()
                    self.interruption_event.clear()
                    continue

                cleaned_chunk = text_chunk.strip()
                if not cleaned_chunk: continue

                if not self.ai_is_speaking.is_set(): self.ai_is_speaking.set()

                audio_chunk, _ = self.voice_generator.generate(cleaned_chunk, speed=settings.SPEED)
                
                if audio_chunk is not None and audio_chunk.size > 0:
                    if self.interruption_event.is_set():
                        while not self.llm_text_queue.empty(): self.llm_text_queue.get()
                        self.ai_is_speaking.clear()
                        self.interruption_event.clear()
                        continue
                    
                    ws.send(audio_chunk.tobytes())

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in AI output thread: {e}")
                self.ai_is_speaking.clear()
                self.interruption_event.clear()
