from .config import settings

class TextChunker:
    """A class to handle intelligent text chunking for voice generation."""

    def __init__(self):
        """Initialize the TextChunker with break points."""
        self.current_text = []
        self.found_first_sentence = False
        self.semantic_breaks = {
            'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
            'while', 'although', 'unless', 'since',
            'and', 'but', 'because', 'then', ',', '-'
        }
        self.sentence_breaks = {'.', '!', '?', ':', ';'}

    def should_process(self, text: str) -> bool:
        """Determines if current accumulated text should be processed."""
        words = text.split()
        current_size = len(words)
        target = settings.FIRST_SENTENCE_SIZE if not self.found_first_sentence else settings.TARGET_SIZE

        if not self.found_first_sentence:
            current_word = words[-1] if words else ""
            has_break = False

            for break_char in self.sentence_breaks:
                if current_word.endswith(break_char):
                    has_break = True
                    break

            if not has_break:
                for break_char in self.semantic_breaks:
                    if break_char == current_word.lower() or \
                       (break_char in [',', '-'] and current_word.endswith(break_char)):
                        has_break = True
                        break

            return has_break

        if any(text.endswith(p) for p in self.sentence_breaks) and current_size > target/2:
            return True

        if current_size > target and words[-1].lower() in self.semantic_breaks:
            return True

        return False

    def process(self, text: str, audio_queue) -> str:
        """Process text chunk and return remaining text."""
        if not text:
            return ""

        words = text.split()
        if not words:
            return ""

        if self.should_process(text):
            if any(c.isalnum() for c in text):
                last_space = text.rfind(' ')
                if last_space > 0:
                    chunk = text[:last_space].strip()
                    remaining = text[last_space:].strip()
                    
                    if chunk:
                        audio_queue.add_sentences([chunk])
                        self.found_first_sentence = True
                    return remaining
                else:
                    audio_queue.add_sentences([text])
                    self.found_first_sentence = True
                    return ""

        return text