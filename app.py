"""
True-Tone: Audio Processing Application
Main entry point for the True-Tone audio processing system.
"""

from ui.dashboard import Dashboard


def main():
    """Initialize and run the True-Tone application."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
