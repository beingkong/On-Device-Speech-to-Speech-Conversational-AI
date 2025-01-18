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

# Setup environment
settings.setup_directories()

def process_input(user_input, messages, generator, speed):
    """Process user input and generate response"""
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Get AI response with interruption check
    print("\nThinking...")
    ai_response = None
    retries = 0
    
    while ai_response is None and retries < settings.MAX_RETRIES:
        # Check for speech while waiting for LLM
        speech_detected, audio_data = check_for_speech()
        if speech_detected:
            print("\nInterrupted during processing!")
            return True, audio_data
            
        ai_response = get_ai_response(
            messages=messages,
            llm_model=settings.LLM_MODEL,
            lm_studio_url=settings.LM_STUDIO_URL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.LM_STUDIO_TEMPERATURE,
            stream=settings.LM_STUDIO_STREAM
        )
        if ai_response is None:
            print("Failed to get response from AI. Retrying...")
            retries += 1
            time.sleep(settings.LM_STUDIO_RETRY_DELAY)
    
    if ai_response is None:
        print("Failed to get AI response after multiple attempts.")
        return False, None
    
    # Filter AI response
    ai_response = filter_response(ai_response)

    # Add AI response to history
    messages.append({"role": "assistant", "content": ai_response})
    print(f"\nAI: {ai_response}")
    
    # Generate and play audio sentence by sentence
    print("\nGenerating and playing speech...")
    sentences = split_into_sentences(ai_response)
    if not sentences:
        return False, None
    
    # Generate and play each sentence with interruption checking
    was_interrupted, interrupt_audio, _ = generate_and_play_sentences(
        sentences=sentences,
        generator=generator,
        speed=speed,
        play_function=play_audio_with_interrupt,
        check_interrupt=check_for_speech,
        output_dir=settings.OUTPUT_DIR
    )
    
    if was_interrupted:
        print("\nInterrupted during playback!")
        return True, interrupt_audio
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
