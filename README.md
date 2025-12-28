BlackForge AI Desktop Agent

A keyboard-first desktop automation dock that connects to a local Ollama server (/api/chat).
On each step, the model must return exactly one JSON action:

type

key

hotkey

scroll

wait

mouse

The agent enforces keyboard-first behavior, can optionally attach screenshots for vision-capable models, and supports Manual approval or Auto-execute.

What the app includes

Always-on-top “glass” dock UI: Start/Stop, Approve/Reject, Settings

Streams actions from Ollama via /api/chat

Optional Vision mode: attaches a base64 PNG screenshot to the request

Keyboard-first enforcement (auto-replans if the model tries to use the mouse)

Settings + logs saved to OS AppData location (via Qt Standard Paths)

PyAutoGUI FAILSAFE enabled: move your mouse to a screen corner to instantly stop

Safety notice

This app can control your computer (typing/clicking).
Use Manual mode first until you trust your prompts and model.

Requirements

Python 3.10+ recommended

Ollama installed and running locally

OS notes (may change over time—check Ollama’s download page if unsure):

Windows: typically Windows 10+

macOS: typically macOS 14 (Sonoma)+

1) Install Ollama
Windows

Download: https://ollama.com/download/windows

Run the installer.

macOS

Download: https://ollama.com/download/mac

Install and launch it.

Linux

Install with one command:

curl -fsSL https://ollama.com/install.sh | sh


Official quickstart/docs:

https://docs.ollama.com/quickstart

2) Start Ollama and verify it works

Check the server is responding:

curl http://localhost:11434/api/tags


Pull a model (example):

ollama pull qwen3-vl:8b


Optional “cloud” model example:

ollama pull qwen3-vl:235b-cloud


List installed models:

ollama list


Notes

If you enable Vision (screenshots), you must select a vision-capable model.

If the model rejects images, the app should retry automatically without images.

3) Run the agent (from source)

Download the code.

Open a terminal in the folder that contains Agent.py.

Upgrade pip:

python -m pip install --upgrade pip


Install dependencies:

pip install requests pyautogui pillow PySide6
pip install pygetwindow pymsgbox pyscreeze pytweening mouseinfo


Run:

python Agent.py

macOS permissions (required)

You must allow control + screenshots:

System Settings → Privacy & Security → Accessibility

System Settings → Privacy & Security → Screen Recording

Linux (Ubuntu/Debian) quick setup
sudo apt update
sudo apt install -y python3-venv python3-pip scrot
python -m pip install --upgrade pip
pip install requests pyautogui pillow PySide6
python Agent.py

4) First-run settings (inside the app)

Open Settings and confirm:

Base URL: http://localhost:11434

Model: choose any installed model

Send screenshots (Vision): ON only if your model supports images

Auto-execute: ON = model has full control

Strict keyboard-first: ON (recommended)

Always confirm mouse actions: ON (recommended)

5) Troubleshooting
“Network error talking to Ollama”

Confirm Ollama is running:

curl http://localhost:11434/api/tags


Confirm the Base URL is exactly:

http://localhost:11434

“Model rejected screenshots / vision unsupported”

Turn Send screenshots (Vision) OFF, or

Select a vision-capable model
