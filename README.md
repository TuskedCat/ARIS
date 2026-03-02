# ARIS — Adaptive Robotic Intelligence System

> **High School Electronics Capstone Project**  
> A 3-DOF robotic arm with a swappable end-effector, controlled via natural language using a local LLM and Kokoro TTS for voice feedback.

---

## What is ARIS?

ARIS is a desktop robotic arm you talk to in plain English.  
Say *"reach forward and grab the block"* and ARIS will:

1. **Understand** — A locally-running Qwen 7B LLM translates your words into a structured JSON command sequence.
2. **Act** — The arm executes the movement over serial/USB.
3. **Respond** — ARIS speaks a natural confirmation using the Kokoro TTS engine.

No cloud. No API keys. Everything runs on-device.

---

## Repo Structure
```
ARIS/
├── ARIS_Terminal.py       # CLI interface — bare-bones, no web server
├── ARIS_Server.py         # Flask web server with full dashboard UI
├── static/
│   └── index.html         # Web dashboard (served by ARIS_Server.py)
├── Output/                # Auto-generated at runtime (gitignored)
│   ├── JSON/              # Saved mission plans
│   ├── WAV/               # Saved TTS audio
│   └── LOG/               # Mission logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Hardware

| Component | Details |
|---|---|
| Degrees of Freedom | 3-DOF movement + end-effector |
| Axes | X (Forward/Back) · Y (Left/Right) · Z (Up/Down) |
| End Effector | Swappable (gripper / tool mount) |
| Controller | *(e.g. Arduino Mega, ESP32)* |
| Communication | *(e.g. USB Serial @ 115200 baud)* |

> ARIS was originally designed as a 6-DOF arm. The current hardware uses 3-DOF Cartesian movement with a modular end-effector. Wrist rotation (`ROTATE_WRIST`) is reserved for a future revision.

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU inference is very slow)
- GGUF model — tested with `qwen2.5-7b-instruct-q5_k_m.gguf`  
  → Download from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
```bash
pip install -r requirements.txt
```

Update the model path in either script:
```python
MODEL_PATH = Path(r"C:\models\qwen\qwen2.5-7b-instruct-q5_k_m.gguf")
```

---

## Running ARIS

**Web dashboard (recommended):**
```bash
python ARIS_Server.py
# → open http://localhost:5000
```

**CLI terminal:**
```bash
python ARIS_Terminal.py
```

---

## Command Reference

The LLM generates these commands automatically — you never write them by hand.

| Command | What it does |
|---|---|
| `MOVE_XYZ(x, y, z)` | Move end-effector in mm |
| `GRAB(true/false)` | Close or open end-effector |
| `HOME()` | Return to rest position |
| `WAIT(seconds)` | Pause execution |
| `ROTATE_WRIST(r,p,y)` | *(Future hardware)* Rotate wrist |

**Distance rule:** 1 Unit = 100 mm for vague prompts like *"reach out a bit."*

---

## How It Works — Two-Pass Inference
```
User Input
    │
    ▼
┌──────────────┐   structured JSON
│  Engineer    │──────────────────►  Movement Sequence → Hardware
│  (Qwen LLM)  │   (thought_chain
└──────────────┘    + commands)
         │
         │  sequence summary
         ▼
┌──────────────┐   natural language
│    Actor     │──────────────────►  Kokoro TTS → WAV
│  (Qwen LLM)  │   personality reply
└──────────────┘
```

The same model runs twice with different system prompts — once as a precise motion planner, once as ARIS's personality layer.

---

## Project Status

- [x] LLM → JSON command pipeline  
- [x] Kokoro TTS voice responses  
- [x] Flask web dashboard  
- [x] CLI terminal interface  
- [x] Mission logging (JSON / WAV / LOG)  
- [ ] Serial communication to arm hardware  
- [ ] Live position telemetry / feedback  
- [ ] ROTATE_WRIST hardware implementation  

---

## Built With

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Local LLM inference  
- [Kokoro TTS](https://github.com/hexgrad/kokoro) — Open-source text-to-speech  
- [Flask](https://flask.palletsprojects.com/) — Web server  
- NumPy + soundfile — Audio processing  

---

*ARIS — High school electronics capstone project.*
