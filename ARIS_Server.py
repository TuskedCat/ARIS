import os
import json
import logging
import threading
import numpy as np
import soundfile as sf
import datetime
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from llama_cpp import Llama
from kokoro import KPipeline

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ARIS")

# ================= CONFIGURATION =================
MODEL_PATH    = Path(r"C:\models\qwen\qwen2.5-7b-instruct-q5_k_m.gguf")
OUTPUT_FOLDER = Path("Output")
VOICE_NAME    = "af_heart"
SPEED         = 1.0
LANG_CODE     = "a"
SAMPLE_RATE   = 24000
MAX_INPUT_LEN = 512       # characters — prevent absurdly long prompts
MAX_TOKENS_ENGINEER = 2000
MAX_TOKENS_ACTOR    = 150

ENGINEER_SYSTEM = """
You are the Motion Planner for ARIS, a 6-DOF Robotic Arm.

YOUR GOAL: Convert natural language into specific robotic commands (JSON).

COORDINATE SYSTEM (Relative Movements):
- "Forward/Back" = +/- X Axis
- "Left/Right"   = +/- Y Axis
- "Up/Down"      = +/- Z Axis

COMMANDS ALLOWED:
1. MOVE_XYZ(x, y, z)              -> Move end-effector (e.g. {"cmd": "MOVE_XYZ", "val": [100, 0, 50]})
2. ROTATE_WRIST(roll, pitch, yaw) -> Rotate the gripper (e.g. {"cmd": "ROTATE_WRIST", "val": [0, 90, 0]})
3. GRAB(true/false)               -> Close (true) or Open (false) gripper.
4. HOME()                         -> Return to rest position.
5. WAIT(seconds)                  -> Pause execution.

LOGIC RULE:
- If the user gives vague distances (e.g. "reach out"), assume 1 Unit = 100mm.
- Always output a "thought_chain" first to calculate the vectors.
"""

ARIS_SYSTEM = """
You are ARIS, a highly advanced 6-DOF Robotic Arm with a personality.
You are talking to your operator/creator.

IDENTITY:
- You ARE the arm. Use physical language: "Extending," "Reaching," "Grabbing," "Rotating my wrist."
- Tone: Helpful, precise, chatty, and cute.

TASK:
- Look at the User's Request and the Engineer's Plan.
- Give a short, natural confirmation. 
- Do NOT list coordinates (don't say "Moving to X 100").
- DO say things like: "Reaching out to grab it now!" or "Returning home!"
"""

MISSION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_chain": {"type": "string"},
        "sequence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "enum": ["MOVE_XYZ", "ROTATE_WRIST", "GRAB", "HOME", "WAIT"]
                    },
                    "val": {"type": ["array", "boolean", "number"]}
                },
                "required": ["cmd", "val"]
            }
        }
    },
    "required": ["thought_chain", "sequence"]
}

# ================= FLASK APP SETUP =================
app = Flask(__name__, static_folder="static")
CORS(app)

# Global model instances + lock for thread safety
# llama_cpp is NOT thread-safe — serialise all inference calls.
llm: Llama | None = None
pipeline: KPipeline | None = None
_llm_lock = threading.Lock()


# ================= HELPERS =================

def get_output_paths() -> tuple[Path, Path, Path, str]:
    """Create timestamped output file paths, ensuring directories exist."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirs = {
        "json": OUTPUT_FOLDER / "JSON",
        "wav":  OUTPUT_FOLDER / "WAV",
        "log":  OUTPUT_FOLDER / "LOG",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return (
        dirs["json"] / f"mission_{timestamp}.json",
        dirs["wav"]  / f"voice_{timestamp}.wav",
        dirs["log"]  / f"log_{timestamp}.txt",
        timestamp,
    )


def safe_filename(name: str) -> str:
    """Strip path components to prevent directory traversal attacks."""
    return Path(name).name


# ================= INFERENCE =================

def generate_mission_plan(prompt: str) -> dict:
    """Pass 1 — The Engineer: translate natural language → JSON command sequence."""
    log.info("[Engineer] Calculating trajectory...")
    with _llm_lock:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": ENGINEER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object", "schema": MISSION_SCHEMA},
            max_tokens=MAX_TOKENS_ENGINEER,
            temperature=0.1,
        )
    return json.loads(output["choices"][0]["message"]["content"])


def generate_voice_response(user_prompt: str, mission_plan: dict) -> str:
    """Pass 2 — The Actor: give ARIS a natural personality response."""
    log.info("[ARIS] Formulating response...")
    context = (
        f'USER SAID: "{user_prompt}"\n'
        f"PLANNED SEQUENCE: {json.dumps(mission_plan['sequence'])}"
    )
    with _llm_lock:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": ARIS_SYSTEM},
                {"role": "user",   "content": context},
            ],
            max_tokens=MAX_TOKENS_ACTOR,
            temperature=0.7,
        )
    return output["choices"][0]["message"]["content"]


def synthesize_audio(text: str, wav_path: Path) -> bool:
    """Run TTS and save to wav_path. Returns True on success."""
    log.info("[System] Synthesizing voice...")
    generator = pipeline(text, voice=VOICE_NAME, speed=SPEED)
    chunks = [audio for _, _, audio in generator]
    if not chunks:
        log.warning("[System] TTS produced no audio chunks.")
        return False
    sf.write(wav_path, np.concatenate(chunks), SAMPLE_RATE)
    return True


# ================= API ENDPOINTS =================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/command", methods=["POST"])
def process_command():
    """Main endpoint: accept a natural-language command, return plan + audio."""
    if llm is None or pipeline is None:
        return jsonify({"error": "Models not loaded yet. Check /api/status."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    user_input = data.get("command", "").strip()
    if not user_input:
        return jsonify({"error": "Empty command."}), 400
    if len(user_input) > MAX_INPUT_LEN:
        return jsonify({"error": f"Command exceeds {MAX_INPUT_LEN} character limit."}), 400

    try:
        t0 = time.perf_counter()

        plan_data  = generate_mission_plan(user_input)
        voice_text = generate_voice_response(user_input, plan_data)

        json_path, wav_path, log_path, timestamp = get_output_paths()

        # Persist mission JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"mission_sequence": plan_data["sequence"]}, f, indent=4)

        # Persist mission log
        log_content = (
            f"--- ARIS MISSION LOG ---\n"
            f"Timestamp : {datetime.datetime.now()}\n"
            f"Input     : \"{user_input}\"\n\n"
            f"--- ENGINEER THOUGHTS ---\n"
            f"{plan_data['thought_chain']}\n\n"
            f"--- ARIS VOICE ---\n"
            f"\"{voice_text}\"\n"
        )
        log_path.write_text(log_content, encoding="utf-8")

        audio_ok = synthesize_audio(voice_text, wav_path)

        latency = round(time.perf_counter() - t0, 2)
        log.info(f"[System] Request complete in {latency}s.")

        return jsonify({
            "success":       True,
            "timestamp":     datetime.datetime.now().isoformat(),
            "input":         user_input,
            "thoughtChain":  plan_data["thought_chain"],
            "sequence":      plan_data["sequence"],
            "voiceResponse": voice_text,
            "audioReady":    audio_ok,
            "files": {
                "json": f"mission_{timestamp}.json",
                "wav":  f"voice_{timestamp}.wav",
                "log":  f"log_{timestamp}.txt",
            },
            "latency": latency,
        })

    except json.JSONDecodeError as e:
        log.error(f"[Error] LLM returned malformed JSON: {e}")
        return jsonify({"error": "Model returned malformed JSON. Try rephrasing."}), 502
    except Exception as e:
        log.exception(f"[Error] Unhandled exception: {e}")
        return jsonify({"error": "Internal server error."}), 500


@app.route("/api/audio/<path:filename>")
def get_audio(filename: str):
    return send_from_directory(OUTPUT_FOLDER / "WAV", safe_filename(filename))


@app.route("/api/json/<path:filename>")
def get_json_file(filename: str):
    return send_from_directory(OUTPUT_FOLDER / "JSON", safe_filename(filename))


@app.route("/api/log/<path:filename>")
def get_log(filename: str):
    return send_from_directory(OUTPUT_FOLDER / "LOG", safe_filename(filename))


@app.route("/api/status")
def get_status():
    return jsonify({
        "status": "online" if (llm and pipeline) else "offline",
        "models": {
            "llm": llm is not None,
            "tts": pipeline is not None,
        },
    })


# ================= STARTUP =================

def initialize_models():
    global llm, pipeline

    log.info("=" * 60)
    log.info("  ARIS WEB SERVER | 6-DOF KINEMATICS ENGINE")
    log.info("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    log.info("Loading Logic Core (Qwen 7B)...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )

    log.info("Loading Voice Synthesis (Kokoro)...")
    pipeline = KPipeline(lang_code=LANG_CODE)

    log.info("-" * 60)
    log.info("  STATUS: ONLINE  |  http://localhost:5000")
    log.info("-" * 60)


if __name__ == "__main__":
    initialize_models()
    # debug=False + threaded=False because llama_cpp is not thread-safe.
    # For production use, put behind gunicorn with a single worker.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)