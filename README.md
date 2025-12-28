# BlackForge AI Desktop Agent

Keyboard-first desktop automation dock that connects to a local **Ollama** server (`/api/chat`). Each step, the model returns **exactly one JSON action** (`type`, `key`, `hotkey`, `scroll`, `wait`, `mouse`). The agent **enforces keyboard-first**, can **attach screenshots** for vision models, and supports **manual approval** or **auto-execute**.

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
- Windows: Windows 10+ required for Ollama
- macOS: macOS 14 (Sonoma)+ required for Ollama

---

## 1) Download + install Ollama

### Windows
1. Download Ollama for Windows: https://ollama.com/download/windows  
2. Run the installer.

### macOS
1. Download Ollama for macOS: https://ollama.com/download/mac  
2. Install and launch it.

### Linux
Install with one command:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama quickstart docs (official): https://docs.ollama.com/quickstart

---

## 2) Start Ollama + verify it works

Verify the server responds:
```bash
curl http://localhost:11434/api/tags
```

Pull a model (example):
```bash
ollama pull qwen3-vl:8b
```

Or if you want a cloud model (faster and more intelligent):
```bash
ollama pull qwen3-vl:235b-cloud
```

List installed models:
```bash
ollama list
```

Notes:
- If you enable Vision (screenshots) in settings, you must choose a vision-capable model.
- If the model rejects images, the app automatically retries without images.

---

## 3) Get the app running (from source)

1. Download the code if you haven’t already.
2. Navigate to where `Agent.py` is located.
3. Open a terminal and run:

Upgrade pip:
```bash
python -m pip install --upgrade pip
```

Install dependencies:
```bash
pip install requests pyautogui pillow PySide6
pip install pygetwindow pymsgbox pyscreeze pytweening mouseinfo
```

Run the agent:
```bash
python Agent.py
```

---

## macOS permissions (required for control + screenshots)

- System Settings → Privacy & Security → Accessibility
- System Settings → Privacy & Security → Screen Recording

---

## Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip scrot
python -m pip install --upgrade pip
pip install requests pyautogui pillow PySide6
python Agent.py
```

---

## 4) First run configuration (in the app)

Open Settings and confirm:

- Base URL: http://localhost:11434
- Model: pick any model you have installed
- Send screenshots (Vision): ON only if your model supports images
- Auto-execute: ON (gives the model full control)
- Strict keyboard-first: ON (recommended)
- Always confirm mouse actions: ON (recommended)

---

## 5) Troubleshooting

### “Network error talking to Ollama”
Check Ollama is running and reachable:
```bash
curl http://localhost:11434/api/tags
```

Confirm Base URL is `http://localhost:11434`.

### “Model rejected screenshots / vision unsupported”
Turn off Send screenshots (Vision), or select a vision-capable model.

The agent will also retry without images automatically.

---

Good luck.
