import unittest

from audio.buffer import AudioBuffer


class AudioBufferTests(unittest.TestCase):
    def test_put_get_fifo_order(self):
        buffer = AudioBuffer(max_size=3)

        buffer.put("first")
        buffer.put("second")

        self.assertEqual(buffer.size(), 2)
        self.assertEqual(buffer.get(), "first")
        self.assertEqual(buffer.get(), "second")
        self.assertIsNone(buffer.get())

    def test_bounded_buffer_drops_oldest_chunk(self):
        buffer = AudioBuffer(max_size=2)

        buffer.put("old")
        buffer.put("middle")
        buffer.put("new")

        self.assertEqual(buffer.size(), 2)
        self.assertEqual(buffer.get(), "middle")
        self.assertEqual(buffer.get(), "new")


if __name__ == "__main__":
    unittest.main()
