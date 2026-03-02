import os
import json
import numpy as np
import soundfile as sf
import datetime
import time
from llama_cpp import Llama
from kokoro import KPipeline

# ================= CONFIGURATION =================
# MODEL SETTINGS
MODEL_PATH = r"C:\models\qwen\qwen2.5-7b-instruct-q5_k_m.gguf" 
OUTPUT_FOLDER = "Output"

# VOICE SETTINGS
VOICE_NAME = 'af_heart'
SPEED = 1.0
LANG_CODE = 'a'

# ROBOT SETTINGS
# We assume the robot starts at (0,0,0) or a "HOME" position.
# The AI will calculate RELATIVE movements from the last known state in its head.
# =================================================

def get_timestamped_paths():
    """Generates unique filenames for saving missions."""
    json_folder = os.path.join(OUTPUT_FOLDER, "JSON")
    wav_folder = os.path.join(OUTPUT_FOLDER, "WAV")
    log_folder = os.path.join(OUTPUT_FOLDER, "LOG")

    for folder in [json_folder, wav_folder, log_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        os.path.join(json_folder, f"mission_{timestamp}.json"),
        os.path.join(wav_folder, f"voice_{timestamp}.wav"),
        os.path.join(log_folder, f"log_{timestamp}.txt")
    )

# --- PASS 1: THE ENGINEER (6-DOF LOGIC) ---
def generate_mission_plan(llm, prompt):
    print("   [Engineer]: Calculating Trajectory...")
    
    ENGINEER_SYSTEM = """
    You are the Motion Planner for ARIS, a 6-DOF Robotic Arm.
    
    YOUR GOAL: Convert natural language into specific robotic commands (JSON).
    
    COORDINATE SYSTEM (Relative Movements):
    - "Forward/Back" = +/- X Axis
    - "Left/Right"   = +/- Y Axis
    - "Up/Down"      = +/- Z Axis
    
    COMMANDS ALLOWED:
    1. MOVE_XYZ(x, y, z)    -> Move end-effector to coordinate (e.g., {"cmd": "MOVE_XYZ", "val": [100, 0, 50]})
    2. ROTATE_WRIST(roll, pitch, yaw) -> Rotate the gripper (e.g., {"cmd": "ROTATE_WRIST", "val": [0, 90, 0]})
    3. GRAB(true/false)     -> Close (true) or Open (false) gripper.
    4. HOME()               -> Return to rest position (0,0,0).
    5. WAIT(seconds)        -> Pause execution.

    LOGIC RULE:
    - If the user gives vague distances (e.g., "reach out"), assume 1 Unit = 100mm (or your preferred scale).
    - Always output a "thought_chain" first to calculate the vectors.
    """

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": ENGINEER_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "thought_chain": {"type": "string"},
                    "sequence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cmd": {"type": "string", "enum": ["MOVE_XYZ", "ROTATE_WRIST", "GRAB", "HOME", "WAIT"]},
                                "val": {
                                    "type": ["array", "boolean", "number"], # Arrays for [x,y,z], bool for Grab
                                } 
                            },
                            "required": ["cmd", "val"]
                        }
                    }
                },
                "required": ["thought_chain", "sequence"]
            }
        },
        max_tokens=2000,
        temperature=0.1 # Keep logic strict!
    )
    return json.loads(output['choices'][0]['message']['content'])

# --- PASS 2: THE ACTOR (ARIS PERSONA) ---
def generate_voice_response(llm, user_prompt, mission_plan):
    print("   [Aris]: formulating response...")
    
    ARIS_SYSTEM = """
    You are ARIS, a highly advanced, 6-DOF Robotic Arm with a personality.
    You are talking to your operator/creator.
    
    IDENTITY:
    - You are physically the arm. Use body language words like "Extending," "Reaching," "Grabbing," "Rotating my wrist."
    - Tone: Helpful, precise, but chatty and cute.
    
    TASK:
    - Look at the User's Request and the Engineer's Plan.
    - Give a short, natural confirmation. 
    - Do NOT list coordinates (don't say "Moving to X 100").
    - DO say things like: "Target acquired! Reaching out to grab it now." or "Returning to home position!"
    """
    
    context_input = f"""
    USER SAID: "{user_prompt}"
    PLANNED SEQUENCE: {json.dumps(mission_plan['sequence'])}
    """

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": ARIS_SYSTEM},
            {"role": "user", "content": context_input}
        ],
        max_tokens=150,
        temperature=0.7 # Creative & Fun
    )
    return output['choices'][0]['message']['content']

# --- MAIN INTERFACE LOOP ---
def start_interface():
    print("\n" + "="*60)
    print("   ARIS SYSTEM BOOT | 6-DOF KINEMATICS ENGINE")
    print("="*60)

    # 1. LOAD MODELS (The "Loading Bar" phase)
    print("-> [System]: Loading Logic Core (Qwen 7B)...")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    
    print("-> [System]: Loading Voice Synthesis (Kokoro)...")
    pipeline = KPipeline(lang_code=LANG_CODE)
    
    print("\n" + "-"*60)
    print("   STATUS: ONLINE.")
    print("   Hello! I am Aris. Ready for calibration or tasking.")
    print("-"*60 + "\n")

    # 2. INTERACTIVE LOOP
    while True:
        try:
            # Get User Input
            user_input = input("[OPERATOR] >> ")
            
            # Exit Conditions
            if user_input.lower() in ["exit", "quit", "shutdown", "sleep"]:
                print("[ARIS]: Powering down servos... Goodnight!")
                break
            
            if not user_input.strip():
                continue

            start_time = time.time()

            # --- STEP A: LOGIC (The Brain) ---
            plan_data = generate_mission_plan(llm, user_input)
            
            # --- STEP B: VOICE (The Soul) ---
            voice_text = generate_voice_response(llm, user_input, plan_data)
            
            # --- STEP C: SAVING ---
            json_path, wav_path, log_path = get_timestamped_paths()
            
            # Save the JSON (The instructions for your future Python Interpreter)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"mission_sequence": plan_data['sequence']}, f, indent=4)
                
            # Save the Logs (For debugging logic)
            log_content = (
                f"--- ARIS MISSION LOG ---\n"
                f"Timestamp: {datetime.datetime.now()}\n"
                f"Input: \"{user_input}\"\n\n"
                f"--- ENGINEER THOUGHTS (IK PLANNER) ---\n"
                f"{plan_data['thought_chain']}\n\n"
                f"--- ARIS VOICE ---\n"
                f"\"{voice_text}\"\n"
            )
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log_content)

            # --- STEP D: AUDIO GENERATION ---
            print("   [System]: Synthesizing Voice...")
            generator = pipeline(voice_text, voice=VOICE_NAME, speed=SPEED)
            all_audio_chunks = []
            for _, _, audio in generator:
                all_audio_chunks.append(audio)

            if all_audio_chunks:
                final_audio = np.concatenate(all_audio_chunks)
                sf.write(wav_path, final_audio, 24000)

            end_time = time.time()
            duration = round(end_time - start_time, 2)

            # --- FEEDBACK DISPLAY ---
            print(f"\n[ARIS]: \"{voice_text}\"")
            print(f"[DATA]: JSON Command sent to {json_path}")
            print(f"[PERF]: Latency {duration}s\n")

        except KeyboardInterrupt:
            print("\n[System]: Emergency Stop Triggered.")
            break
        except Exception as e:
            print(f"\n[ERROR]: Logic Failure: {e}")

if __name__ == "__main__":
    start_interface()