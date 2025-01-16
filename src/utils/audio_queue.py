from queue import Queue
import threading
import time
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path
from .audio_io import save_audio_file

class AudioGenerationQueue:
    def __init__(self, generator, speed: float = 1.0, output_dir: Optional[Path] = None):
        self.generator = generator
        self.speed = speed
        self.output_dir = output_dir
        self.sentence_queue = Queue()  # Queue for incoming sentences
        self.audio_queue = Queue()     # Queue for generated audio segments
        self.is_running = False
        self.generation_thread = None
        
    def start(self):
        """Start the audio generation thread"""
        if not self.is_running:
            self.is_running = True
            self.generation_thread = threading.Thread(target=self._generation_worker)
            self.generation_thread.daemon = True  # Thread will exit when main program exits
            self.generation_thread.start()
            
    def stop(self):
        """Stop the audio generation thread"""
        self.is_running = False
        if self.generation_thread:
            self.generation_thread.join()
            self.generation_thread = None
            
    def add_sentences(self, sentences: List[str]):
        """Add sentences to the generation queue"""
        for sentence in sentences:
            if sentence.strip():  # Skip empty sentences
                self.sentence_queue.put(sentence)
            
    def get_next_audio(self) -> Tuple[Optional[np.ndarray], Optional[Path]]:
        """Get the next generated audio segment, non-blocking"""
        try:
            return self.audio_queue.get_nowait()
        except:
            return None, None
            
    def clear_queues(self):
        """Clear both sentence and audio queues"""
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except:
                pass
                
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
                
    def _generation_worker(self):
        """Worker thread that continuously generates audio from sentences"""
        while self.is_running:
            try:
                # Try to get a sentence from the queue
                try:
                    sentence = self.sentence_queue.get_nowait()
                except:
                    time.sleep(0.01)  # Short sleep if no sentences
                    continue
                    
                # Generate audio for the sentence
                try:
                    audio_data, _ = self.generator.generate(sentence, speed=self.speed)
                    
                    # Save audio file if output directory is specified
                    output_path = None
                    if self.output_dir:
                        output_path = save_audio_file(audio_data, self.output_dir)
                    
                    # Add to audio queue
                    self.audio_queue.put((audio_data, output_path))
                    
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error in generation worker: {str(e)}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop 