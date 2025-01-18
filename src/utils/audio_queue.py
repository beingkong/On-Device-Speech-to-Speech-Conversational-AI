from queue import Queue
import threading
import time
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from .audio_io import save_audio_file

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_generation.log'),
        logging.StreamHandler()
    ]
)

class AudioGenerationQueue:
    def __init__(self, generator, speed: float = 1.0, output_dir: Optional[Path] = None):
        self.generator = generator
        self.speed = speed
        self.output_dir = output_dir or Path("generated_audio")
        self.output_dir.mkdir(exist_ok=True)
        self.sentence_queue = Queue()  # Queue for incoming sentences
        self.audio_queue = Queue()     # Queue for generated audio segments
        self.is_running = False
        self.generation_thread = None
        self.sentences_processed = 0
        self.audio_generated = 0
        self.failed_sentences = []
        
    def start(self):
        """Start the audio generation thread"""
        if not self.is_running:
            self.is_running = True
            self.generation_thread = threading.Thread(target=self._generation_worker)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            logging.info("Audio generation thread started")
            
    def stop(self):
        """Stop the audio generation thread"""
        if self.generation_thread:
            logging.info("Stopping audio generation thread - waiting for remaining sentences...")
            
            # Wait for all sentences to be processed
            while not self.sentence_queue.empty():
                time.sleep(0.1)  # Small sleep while waiting
                
            # Give a small buffer time for the last sentence to finish processing
            time.sleep(0.5)
            
            self.is_running = False
            self.generation_thread.join()
            self.generation_thread = None
            
            logging.info(f"Audio generation thread stopped. Summary:")
            logging.info(f"- Total sentences processed: {self.sentences_processed}")
            logging.info(f"- Successfully generated audio files: {self.audio_generated}")
            logging.info(f"- Failed sentences: {len(self.failed_sentences)}")
            if self.failed_sentences:
                logging.info("Failed sentences:")
                for i, (sentence, error) in enumerate(self.failed_sentences, 1):
                    logging.info(f"{i}. Sentence: {sentence[:100]}... Error: {error}")
                    
    def add_sentences(self, sentences: List[str]):
        """Add sentences to the generation queue"""
        added_count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                self.sentence_queue.put(sentence)
                logging.debug(f"Added sentence to queue: {sentence}")
                added_count += 1
                
        queue_size = self.sentence_queue.qsize()
        logging.info(f"Added {added_count} sentences to queue. Current queue size: {queue_size}")
        
        # Start the thread if it's not running
        if not self.is_running:
            self.start()
            
    def get_next_audio(self):
        """Get the next generated audio segment, non-blocking"""
        try:
            audio_data, output_path = self.audio_queue.get_nowait()
            logging.debug(f"Retrieved audio from queue: {output_path}")
            return audio_data, output_path
        except:
            return None, None
            
    def clear_queues(self):
        """Clear both sentence and audio queues"""
        sentences_cleared = 0
        audio_cleared = 0
        
        while not self.sentence_queue.empty():
            try:
                sentence = self.sentence_queue.get_nowait()
                sentences_cleared += 1
                logging.debug(f"Cleared sentence from queue: {sentence[:100]}...")
            except:
                pass
                
        while not self.audio_queue.empty():
            try:
                _, path = self.audio_queue.get_nowait()
                audio_cleared += 1
                logging.debug(f"Cleared audio from queue: {path}")
            except:
                pass
        
        logging.info(f"Cleared queues: {sentences_cleared} sentences and {audio_cleared} audio segments removed")
                
    def _generation_worker(self):
        """Worker thread that continuously generates audio from sentences"""
        while self.is_running or not self.sentence_queue.empty():  # Changed condition to process remaining sentences
            try:
                # Try to get a sentence from the queue
                try:
                    sentence = self.sentence_queue.get_nowait()
                    self.sentences_processed += 1
                    logging.info(f"Processing sentence {self.sentences_processed}: {sentence}")
                except:
                    if not self.is_running and self.sentence_queue.empty():
                        break  # Exit if we're stopping and queue is empty
                    time.sleep(0.01)  # Short sleep if no sentences
                    continue
                    
                # Generate audio for the sentence
                try:
                    logging.debug(f"Generating audio for: {sentence}")
                    audio_data, phonemes = self.generator.generate(sentence, speed=self.speed)
                    
                    if audio_data is None or len(audio_data) == 0:
                        raise ValueError("Generated audio data is empty")
                        
                    # Save audio file
                    output_path = save_audio_file(audio_data, self.output_dir)
                    self.audio_generated += 1
                    logging.info(f"Successfully generated audio {self.audio_generated} for sentence: {sentence}")
                    logging.info(f"Saved audio to: {output_path}")
                    
                    # Add to audio queue
                    self.audio_queue.put((audio_data, output_path))
                    
                except Exception as e:
                    error_msg = str(e)
                    self.failed_sentences.append((sentence, error_msg))
                    logging.error(f"Error generating audio for sentence: {sentence}")
                    logging.error(f"Error details: {error_msg}")
                    continue
                    
            except Exception as e:
                logging.error(f"Error in generation worker: {str(e)}")
                if not self.is_running and self.sentence_queue.empty():
                    break  # Exit if we're stopping and queue is empty
                time.sleep(0.1)  # Sleep on error to prevent tight loop 