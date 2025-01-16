import os
import traceback
import time
from src.utils.config import settings
# Local imports
from src.utils import (
    play_audio, VoiceGenerator, split_into_sentences,
    filter_response, get_ai_response, generate_and_play_sentences,
    handle_commands
)

def main():
    # Initialize the voice generator
    generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
    
    try:
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
        print("Type your message and press Enter. Type 'quit' to exit.")
        print("Available commands:")
        print("  speed=X : Change speech speed (e.g., speed=1.2)")
        print("  voice=X : Change voice (e.g., voice=af_sky)")
        print("  voices  : List available voices")
        print("  mix=voice1,voice2[:weight1,weight2] : Mix two voices")
        print("-" * 50)
        
        # Initialize chat history
        messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        speed = settings.SPEED
        
        # Chat loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle commands
                if handle_commands(user_input, generator, speed, settings.TTS_MODEL):
                    if user_input.lower() == 'quit':
                        print("Goodbye!")
                        break
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Add user message to history
                messages.append({"role": "user", "content": user_input})
                
                # Get AI response
                print("\nThinking...")
                ai_response = get_ai_response(
                    messages=messages,
                    llm_model=settings.LLM_MODEL,
                    lm_studio_url=settings.LM_STUDIO_URL,
                    max_tokens=settings.MAX_TOKENS
                )
                if ai_response is None:
                    print("Failed to get response from AI. Please try again.")
                    continue
                
                # Filter AI response
                ai_response = filter_response(ai_response)

                # Add AI response to history
                messages.append({"role": "assistant", "content": ai_response})
                print(f"\nAI: {ai_response}")
                
                # Generate and play audio
                print("\nGenerating and playing speech...")
                sentences = split_into_sentences(ai_response)
                if not sentences:
                    continue
                
                # Generate and play each sentence
                _, _, _ = generate_and_play_sentences(
                    sentences=sentences,
                    generator=generator,
                    speed=speed,
                    play_function=play_audio,
                    output_dir=settings.OUTPUT_DIR
                )
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 