import time
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import numpy as np
from .audio_io import save_audio_file, play_audio
from .audio_queue import AudioGenerationQueue

def generate_and_play_sentences(
    sentences: List[str],
    generator,
    speed: float = 1.0,
    play_function: Callable = play_audio,
    check_interrupt: Optional[Callable] = None,
    output_dir: Optional[Path] = None,
    sample_rate: Optional[int] = None
) -> Tuple[bool, Optional[np.ndarray], List[Path]]:
    """Generate and play audio for each sentence with optional interruption checking"""
    
    # Initialize audio queue
    audio_queue = AudioGenerationQueue(generator, speed, output_dir)
    audio_queue.start()
    
    # Add all sentences to the queue
    audio_queue.add_sentences(sentences)
    
    # Keep track of generated audio files
    audio_files = []
    was_interrupted = False
    interrupt_audio = None
    
    try:
        while True:
            # Check for interruption if function provided
            if check_interrupt:
                interrupted, audio_data = check_interrupt()
                if interrupted:
                    was_interrupted = True
                    interrupt_audio = audio_data
                    break
            
            # Try to get next generated audio
            audio_data, output_path = audio_queue.get_next_audio()
            
            if audio_data is not None:
                # Play the audio
                if output_path:
                    audio_files.append(output_path)
                
                # Play audio with provided function
                if play_function:
                    try:
                        was_interrupted, interrupt_data = play_function(audio_data, sample_rate) if sample_rate else play_function(audio_data)
                        if was_interrupted:
                            interrupt_audio = interrupt_data
                            break
                    except Exception as e:
                        print(f"Error playing audio: {str(e)}")
                        continue
            
            # Check if we're done (no more sentences and queue is empty)
            if audio_queue.sentence_queue.empty() and audio_queue.audio_queue.empty():
                break
                
            # Small sleep to prevent tight loop
            time.sleep(0.01)
            
    except Exception as e:
        print(f"Error in generate_and_play_sentences: {str(e)}")
    finally:
        # Clean up
        audio_queue.stop()
        
    return was_interrupted, interrupt_audio, audio_files 