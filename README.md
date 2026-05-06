# True-Tone

True-Tone is an advanced audio processing application designed for real-time audio capture, processing, and visualization from both microphone and system audio sources.

## Project Structure

```
true-tone/
├── app.py                 # Main application entry point
├── audio/                 # Audio processing modules
│   ├── mic_capture.py     # Microphone capture functionality
│   ├── system_capture.py  # System audio capture
│   ├── buffer.py          # Audio buffer management
│   └── wav_utils.py       # WAV file utilities
├── ui/                    # User interface modules
│   └── dashboard.py       # Main dashboard interface
├── test_chunks/           # Test audio files directory
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Features

- Real-time microphone audio capture
- System audio capture and processing
- Audio buffering and queue management
- WAV file I/O utilities
- Interactive dashboard UI
- Audio visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd true-tone
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app.py
```

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]