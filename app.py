"""HuggingFace Spaces entry point for Solar Intelligence Platform."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "panel", "serve",
        "src/solar_intelligence/ui/panel_dashboard.py",
        "--address=0.0.0.0",
        "--port=7860",
        "--allow-websocket-origin=*",
        "--prefix=/",
    ])
