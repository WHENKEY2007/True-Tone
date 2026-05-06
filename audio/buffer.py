"""
Audio buffer management module for True-Tone.
Handles buffering and queue management for audio streams.
"""

from collections import deque
import threading


class AudioBuffer:
    """Thread-safe audio buffer for managing audio chunks."""
    
    def __init__(self, max_size=100):
        """
        Initialize audio buffer.
        
        Args:
            max_size: Maximum number of chunks to buffer
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def put(self, chunk):
        """Add audio chunk to buffer."""
        with self.lock:
            self.buffer.append(chunk)
    
    def get(self):
        """Retrieve audio chunk from buffer."""
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None
    
    def is_empty(self):
        """Check if buffer is empty."""
        with self.lock:
            return len(self.buffer) == 0
    
    def size(self):
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
