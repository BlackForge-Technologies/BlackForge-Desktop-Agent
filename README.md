# BlackForge AI Desktop Agent

Keyboard-first desktop automation dock that connects to a local **Ollama** server (`/api/chat`). Each step, the model returns **exactly one JSON action** (type/key/hotkey/scroll/wait/mouse). The agent **enforces keyboard-first**, can **attach screenshots** for vision models, and supports **manual approval** or **auto-execute**.

---

## What this app does

- Floating always-on-top “glass” dock UI (Start/Stop, Approve/Reject, Settings)
- Streams actions from Ollama `/api/chat`
- Optional screenshot “Vision” mode (base64 PNG attached to the chat request)
- Strict keyboard-first enforcement (auto-replans if the model chooses mouse)
- Settings + logs stored in OS AppData location (Qt Standard Paths)
- PyAutoGUI FAILSAFE enabled (move mouse to a screen corner to stop)

---

## Safety notice

This app can type/click on your computer. Start in **Manual** mode until you trust your prompt + model.

---

## Requirements

- Python 3.10+ recommended
- Ollama installed and running on the same machine
- Windows: Windows 10+ required for Ollama :contentReference[oaicite:0]{index=0}
- macOS: macOS 14 (Sonoma)+ required for Ollama :contentReference[oaicite:1]{index=1}

---

## 1) Download + install Ollama

### Windows
1. Download Ollama for Windows: https://ollama.com/download/windows :contentReference[oaicite:2]{index=2}
2. Run the installer.

### macOS
1. Download Ollama for macOS: https://ollama.com/download/mac :contentReference[oaicite:3]{index=3}
2. Install and launch it.

### Linux
Install with one command: :contentReference[oaicite:4]{index=4}
'''''
curl -fsSL https://ollama.com/install.sh | sh
.....

Ollama quickstart docs (official): https://docs.ollama.com/quickstart 
Ollama Documentation

2) Start Ollama + verify it works
Verify the server responds:

'''''
curl http://localhost:11434/api/tags
.....

   
Pull a model (example):

'''''
ollama pull qwen3-vl:8b
.....


Or if you want a cloud model(Faster and more intelligent):

'''''
ollama pull qwen3-vl:235b-cloud
.....


List installed models:

'''''
ollama list
.....


Notes:

If you enable Vision (screenshots) in settings, you must choose a vision-capable model.

If the model rejects images, the app automatically retries without images.

3) Get the app running (from source)
   Download the code if you haven't allready.
   navigate to where Agent.py is located.

   open command terminal and type the following:

   '''''
   python -m pip install --upgrade pip
   .....


   '''''
   pip install requests pyautogui pillow PySide6
   .....


   '''''
   pip install pygetwindow pymsgbox pyscreeze pytweening mouseinfo
   .....

  Then you can run the Agent with one command: 

  '''''
  Python Agent.py
  .....

macOS permissions (required for control + screenshots):

System Settings → Privacy & Security → Accessibility

System Settings → Privacy & Security → Screen Recording

Linux (Ubuntu/Debian)

'''''
sudo apt update
.....


'''''
sudo apt install -y python3-venv python3-pip scrot
.....


'''''
python -m pip install --upgrade pip
.....


'''''
pip install requests pyautogui pillow PySide6
.....


Then you can run the Agent with one command:

'''''
python Agent.py
.....



4) First run configuration (in the app)
Open Settings and confirm:

Base URL: http://localhost:11434

Model: pick any model you have installed. 

Send screenshots (Vision): ON only if your model supports images

Auto-execute: ON (Gives the model full controll)

Strict keyboard-first: ON (recommended, if mouse actions are needed then turn off. Windows doesen't need)

Always confirm mouse actions: ON (recommended)

5) Troubleshooting
“Network error talking to Ollama”
Check Ollama is running and reachable:

'''''
curl http://localhost:11434/api/tags
.....


'''''
Confirm Base URL is http://localhost:11434
.....



“Model rejected screenshots / vision unsupported”
Turn off Send screenshots (Vision), or select a vision-capable model.

The agent will also retry without images automatically.


~Good Luck!
