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
```bash
curl -fsSL https://ollama.com/install.sh | sh
Ollama quickstart docs (official): https://docs.ollama.com/quickstart 
Ollama Documentation

2) Start Ollama + verify it works
Verify the server responds:

bash
Copy code
curl http://localhost:11434/api/tags
Pull a model (example):

bash
Copy code
ollama pull qwen2.5:7b
List installed models:

bash
Copy code
ollama list
Notes:

If you enable Vision (screenshots) in settings, you must choose a vision-capable model.

If the model rejects images, the app automatically retries without images.

3) Get the app running (from source)
Save the file
Save your Python file as:

blackforge_agent.py

Create a virtual environment + install dependencies
Windows (PowerShell)
powershell
Copy code
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install requests pyautogui pillow PySide6
python .\blackforge_agent.py
If you hit PyAutoGUI dependency errors:

powershell
Copy code
pip install pygetwindow pymsgbox pyscreeze pytweening mouseinfo
macOS (Terminal)
bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install requests pyautogui pillow PySide6
python blackforge_agent.py
macOS permissions (required for control + screenshots):

System Settings → Privacy & Security → Accessibility

System Settings → Privacy & Security → Screen Recording

Linux (Ubuntu/Debian)
bash
Copy code
sudo apt update
sudo apt install -y python3-venv python3-pip scrot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install requests pyautogui pillow PySide6
python blackforge_agent.py
4) First run configuration (in the app)
Open Settings and confirm:

Base URL: http://localhost:11434

Model: pick one from “Refresh models”

Send screenshots (Vision): ON only if your model supports images

Auto-execute: OFF (recommended initially)

Strict keyboard-first: ON (recommended)

Always confirm mouse actions: ON (recommended)

5) Troubleshooting
“Network error talking to Ollama”
Check Ollama is running and reachable:

bash
Copy code
curl http://localhost:11434/api/tags
Confirm Base URL is http://localhost:11434

“Model rejected screenshots / vision unsupported”
Turn off Send screenshots (Vision), or select a vision-capable model.

The agent will also retry without images automatically.

PyAutoGUI FAILSAFE triggered
You moved the mouse to a screen corner (intentional safety stop). Restart the agent.

6) Optional: Build an executable (PyInstaller)
Install PyInstaller:

bash
Copy code
pip install pyinstaller
Build one-file app:

bash
Copy code
pyinstaller --noconfirm --onefile --windowed --name BlackForge blackforge_agent.py
Output:

Windows: dist/BlackForge.exe

macOS/Linux: dist/BlackForge

macOS: you may need to grant Accessibility + Screen Recording to the built app as well.

