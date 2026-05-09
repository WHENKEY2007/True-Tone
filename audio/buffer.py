"""
Audio buffer management module for True-Tone.
Handles buffering and queue management for audio streams.
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Any


class AudioBuffer:
    """Thread-safe audio buffer for managing audio chunks.

    Provides put/get semantics with optional peek (non-consuming read)
    and bulk drain operations for the pipeline and UI.

    Args:
        max_size: Maximum number of chunks to store. Oldest chunks
            are silently dropped when the buffer is full.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: deque[Any] = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def put(self, chunk: Any) -> None:
        """Add audio chunk to buffer."""
        with self.lock:
            self.buffer.append(chunk)

    def get(self) -> Any | None:
        """Retrieve and remove the oldest audio chunk from the buffer.

        Returns:
            The oldest chunk, or None if the buffer is empty.
        """
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None

    def peek(self) -> Any | None:
        """Read the oldest chunk without consuming it.

        Useful for waveform display where you want to visualize
        the current chunk without removing it from the queue.

        Returns:
            The oldest chunk, or None if the buffer is empty.
        """
        with self.lock:
            if self.buffer:
                return self.buffer[0]
            return None

    def peek_latest(self) -> Any | None:
        """Read the newest chunk without consuming it.

        Returns:
            The most recently added chunk, or None if empty.
        """
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
            return None

    def get_all(self) -> list[Any]:
        """Drain the entire buffer and return all chunks as a list.

        The buffer will be empty after this call.

        Returns:
            A list of all buffered chunks (oldest first).
        """
        with self.lock:
            items = list(self.buffer)
            self.buffer.clear()
            return items

    def get_latest(self, n: int) -> list[Any]:
        """Return the most recent *n* chunks without consuming them.

        Args:
            n: Number of recent chunks to return.

        Returns:
            A list of up to *n* chunks (oldest first).
        """
        with self.lock:
            items = list(self.buffer)
            return items[-n:] if n < len(items) else items

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self.lock:
            return len(self.buffer) == 0

    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)

    def clear(self) -> None:
        """Remove all items from the buffer."""
        with self.lock:
            self.buffer.clear()
