import os
import msvcrt
import traceback
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.utils.config import settings
from src.utils import (
    VoiceGenerator, split_into_sentences, filter_response,
    get_ai_response, play_audio_with_interrupt, handle_commands,
    init_vad_pipeline, detect_speech_segments, record_audio,
    record_continuous_audio, check_for_speech, transcribe_audio,
    generate_and_play_sentences
)
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.llm import parse_stream_chunk
import threading

# Setup environment
settings.setup_directories()

def process_input(user_input, messages, generator, speed):
    """Process user input and generate response"""
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Get AI response with interruption check and streaming
    print("\nThinking...")
    ai_response = []
    current_sentence = []
    complete_response = []
    
    try:
        response_stream = get_ai_response(
            messages=messages,
            llm_model=settings.LLM_MODEL,
            lm_studio_url=settings.LM_STUDIO_URL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.LM_STUDIO_TEMPERATURE,
            stream=True  # Enable streaming
        )
        
        if response_stream is None:
            print("Failed to get AI response stream.")
            return False, None
            
        # Start audio generation queue early
        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        
        # Start a thread for audio playback
        def audio_playback_worker():
            was_interrupted = False
            interrupt_audio = None
            
            try:
                while True:
                    # Check for interruption
                    speech_detected, audio_data = check_for_speech()
                    if speech_detected:
                        was_interrupted = True
                        interrupt_audio = audio_data
                        break
                    
                    # Try to get and play next audio segment
                    audio_data, _ = audio_queue.get_next_audio()
                    if audio_data is not None:
                        was_interrupted, interrupt_data = play_audio_with_interrupt(audio_data)
                        if was_interrupted:
                            interrupt_audio = interrupt_data
                            break
                    else:
                        # No audio available yet, small sleep
                        time.sleep(0.01)
                        
                    # Check if we should exit (text generation done and queue empty)
                    if not audio_queue.is_running and audio_queue.sentence_queue.empty() and audio_queue.audio_queue.empty():
                        break
                        
            except Exception as e:
                print(f"Error in audio playback: {str(e)}")
                
            return was_interrupted, interrupt_audio
            
        # Start audio playback thread
        playback_thread = threading.Thread(target=audio_playback_worker)
        playback_thread.daemon = True
        playback_thread.start()
        
        # Process streaming response
        for chunk in response_stream:
            # Parse the chunk
            data = parse_stream_chunk(chunk)
            if not data:
                continue
                
            # Get the text from the chunk
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                
                # Get content from delta if present
                if "delta" in choice:
                    delta = choice["delta"]
                    if "content" in delta:
                        content = delta["content"]
                        if content:
                            print(content, end='', flush=True)
                            current_sentence.append(content)
                            
                            # Check if this content completes a sentence
                            text = ''.join(current_sentence)
                            # Look for sentence endings or line breaks
                            sentence_end = -1
                            for i, char in enumerate(text):
                                if char in '.!?' or char == '\n':
                                    sentence_end = i + 1
                            
                            if sentence_end > 0:
                                # Extract the complete sentence
                                sentence = text[:sentence_end].strip()
                                if sentence:
                                    ai_response.append(sentence)
                                    complete_response.append(sentence)
                                    audio_queue.add_sentences([sentence])
                                # Keep the remaining text
                                current_sentence = [text[sentence_end:]]
                
                # Check for stream completion
                if choice.get("finish_reason") == "stop":
                    # Process any remaining text as the final sentence
                    text = ''.join(current_sentence).strip()
                    if text:
                        # Clean up any final special characters
                        text = text.rstrip('.,!?')  # Remove trailing punctuation
                        if text:  # If there's still text after cleanup
                            ai_response.append(text)
                            complete_response.append(text)
                            audio_queue.add_sentences([text])
                    break  # Exit the stream processing
        
        # Add complete response to history
        messages.append({"role": "assistant", "content": ' '.join(complete_response)})
        print()  # New line after streaming
        
        # Wait for audio generation and playback to finish
        time.sleep(0.1)  # Small delay to ensure final sentence is queued
        audio_queue.stop()  # Signal generation to stop after current sentence
        playback_thread.join()  # Wait for playback to finish
        
        return False, None
        
    except Exception as e:
        print(f"\nError during streaming: {str(e)}")
        if 'audio_queue' in locals():
            audio_queue.stop()
        return False, None

def main():
    try:
        # Initialize the voice generator
        generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
        
        # Initialize Whisper
        print("\nInitializing Whisper model...")
        whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL)
        
        # Initialize VAD pipeline
        print("\nInitializing Voice Activity Detection...")
        vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
        
        print("\n=== Voice Chat Bot Initializing ===")
        print("Device being used:", generator.device)
        
        # Initialize the model
        print("\nInitializing voice generator...")
        result = generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
        print(result)
        
        # Test LM Studio connection
        print("\nTesting LM Studio connection...")
        test_response = get_ai_response(
            messages=[{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                     {"role": "user", "content": "Hello"}],
            llm_model=settings.LLM_MODEL,
            lm_studio_url=settings.LM_STUDIO_URL,
            max_tokens=settings.MAX_TOKENS
        )
        if test_response is None:
            print("Error: Could not connect to LM Studio. Make sure it's running and the API is accessible.")
            return
        
        print("\n=== Voice Chat Bot Ready ===")
        print("The bot is now listening for speech.")
        print("Just start speaking, and I'll respond automatically!")
        print("You can interrupt me anytime by starting to speak.")
        print("\nOther options:")
        print("  - Type text messages and press Enter")
        print("  - Use 'v' for manual voice recording")
        print("  - Commands: speed=X, voice=X, voices, mix=voice1,voice2")
        print("  - Type 'quit' to exit")
        print("-" * 50)
        
        # Initialize chat history
        messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        speed = settings.SPEED
        
        # Chat loop
        while True:
            try:
                # Check for keyboard input (non-blocking)
                if msvcrt.kbhit():
                    user_input = input("\nYou (text): ").strip()
                    
                    # Check for exit command
                    if user_input.lower() == 'quit':
                        print("Goodbye!")
                        break
                        
                    # Handle manual voice recording
                    if user_input.lower() == 'v':
                        audio_data = record_audio()
                        if audio_data is not None:
                            speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                            if speech_segments is not None:
                                print("\nTranscribing recorded speech...")
                                user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                if user_input.strip():
                                    print(f"You (voice): {user_input}")
                                    was_interrupted, speech_data = process_input(user_input, messages, generator, speed)
                                    if was_interrupted and speech_data is not None:
                                        # Process the speech that caused interruption
                                        speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                                        if speech_segments is not None:
                                            print("\nTranscribing interrupted speech...")
                                            user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                            if user_input.strip():
                                                print(f"You (voice): {user_input}")
                                                process_input(user_input, messages, generator, speed)
                    else:
                        # Handle other commands
                        if handle_commands(user_input, generator, speed, settings.TTS_MODEL):
                            continue
                        # Process text input with interruption handling
                        was_interrupted, speech_data = process_input(user_input, messages, generator, speed)
                        if was_interrupted and speech_data is not None:
                            # Process the speech that caused interruption
                            speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                            if speech_segments is not None:
                                print("\nTranscribing interrupted speech...")
                                user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                if user_input.strip():
                                    print(f"You (voice): {user_input}")
                                    process_input(user_input, messages, generator, speed)
                        continue
                
                # Continuously monitor audio
                audio_data = record_continuous_audio()
                if audio_data is not None:
                    # Detect speech segments
                    speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                    
                    if speech_segments is not None:
                        print("\nTranscribing detected speech...")
                        user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                        if user_input.strip():
                            print(f"You (voice): {user_input}")
                            # Process the transcribed input with interruption handling
                            was_interrupted, speech_data = process_input(user_input, messages, generator, speed)
                            if was_interrupted and speech_data is not None:
                                # Process the speech that caused interruption
                                speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                                if speech_segments is not None:
                                    print("\nTranscribing interrupted speech...")
                                    user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                    if user_input.strip():
                                        print(f"You (voice): {user_input}")
                                        process_input(user_input, messages, generator, speed)
                    else:
                        print("No clear speech detected, please try again.")
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
