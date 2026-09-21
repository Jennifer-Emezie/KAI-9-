#!/usr/bin/env python3
"""
Tier 1 - Master Brain (Runs on Laptop / RTX 5070 Ti)
Utilizes LangGraph Send API to spawn sequential agents. 
- Integrated Native Windows BLE Heart Rate Monitor
- Connects directly to Pi 4 (Tier 3) for Camera Snapshots
- Connects to local Nav Engine (Tier 2) for Movement
- Active Listening Window & Memory Summarization every 6 prompts
"""

import asyncio
import zmq
import zmq.asyncio
from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import operator
import pyttsx3
import threading
import sys
import re
import time

# Import your local STT module
from audio_stt import start_stt_engine
from bleak import BleakClient, BleakScanner

PI4_IP = "10.26.191.91" # TODO: REPLACE WITH YOUR PI 4's ACTUAL IP ADDRESS

# --- MEMORY SYSTEM ---
memory_summary = "No long-term memories yet."
conversation_log = []
prompt_counter = 0

async def trigger_memory_compaction():
    """Compresses the conversation log."""
    global memory_summary, conversation_log, prompt_counter
    
    print("\n[MEMORY] 6-prompt cycle reached. Compressing short-term memory...")
    log_text = "\n".join(conversation_log)
    
    prompt = (
        "You are a memory consolidation engine. Extract the core facts, user preferences, "
        "and important context from the following conversation log. Combine it seamlessly "
        f"with the existing memory summary: '{memory_summary}'. Keep it concise.\n\nLog:\n{log_text}"
    )
    
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    memory_summary = res.content
    conversation_log.clear()
    prompt_counter = 0
    print(f"[MEMORY] New Memory Summary: {memory_summary}\n")

# --- BLUETOOTH HEART RATE (LAPTOP SIDE) ---
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HRM_DEVICE_ADDRESS = "E2:BF:84:6C:13:68"
HRM_DEVICE_NAME    = "HRM BAND 41244-17"

latest_heart_rate = "unknown"
ble_connected = False

def parse_heart_rate(data: bytearray) -> int:
    return int.from_bytes(data[1:3], byteorder="little") if data[0] & 0x01 else data[1]

def hr_notification_handler(sender, data):
    global latest_heart_rate
    latest_heart_rate = parse_heart_rate(data)

def is_target_band(device) -> bool:
    if device.address and device.address.upper() == HRM_DEVICE_ADDRESS.upper(): return True
    if device.name and HRM_DEVICE_NAME.lower() in device.name.lower(): return True
    return False

async def ble_heart_rate_loop():
    global latest_heart_rate, ble_connected
    while True:
        devices = await BleakScanner.discover(timeout=10.0, service_uuids=[HR_SERVICE_UUID])
        target_device = next((d for d in devices if is_target_band(d)), None)

        if not target_device:
            ble_connected, latest_heart_rate = False, "unknown"
            await asyncio.sleep(10)
            continue

        try:
            disconnected_event = asyncio.Event()
            def on_disconnect(client): disconnected_event.set()

            async with BleakClient(target_device, disconnected_callback=on_disconnect) as client:
                ble_connected = True
                await client.start_notify(HR_MEASUREMENT_UUID, hr_notification_handler)
                await disconnected_event.wait()
        except Exception:
            pass
        finally:
            ble_connected, latest_heart_rate = False, "unknown"
            await asyncio.sleep(10)

# --- AUDIO SETUP & ACTIVE LISTENING ---
tts_lock = threading.Lock()
is_speaking = False
expecting_response = False
response_timer = None

def close_listening_window():
    global expecting_response
    if expecting_response:
        print("\n[AUDIO] Listening window closed.")
        expecting_response = False

def open_listening_window():
    global expecting_response, response_timer, main_loop
    expecting_response = True
    print("\n[AUDIO] Kai is waiting for your response (8 seconds)...")
    if response_timer:
        response_timer.cancel()
    response_timer = main_loop.call_later(8.0, close_listening_window)

def _speak_thread_target(text: str):
    global is_speaking
    with tts_lock:
        is_speaking = True
        try:
            import pythoncom
            pythoncom.CoInitialize() 
        except ImportError:
            pass
            
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")
        finally:
            # FIX: Wait 0.5s before enabling STT to let the room echo die down!
            # This stops the robot from hearing its own voice and looping.
            time.sleep(0.5)
            is_speaking = False
        
        # Check for common question keywords to open mic
        question_words = ["how", "what", "where", "why", "are", "do", "can", "is"]
        clean_text = text.lower().strip()
        if clean_text.endswith("?") or any(clean_text.startswith(w) for w in question_words):
            print("[AUDIO] Detected question or intent. Opening mic.")
            main_loop.call_soon_threadsafe(open_listening_window)

def speak_text(text: str):
    print(f"\n[TTS] Speaking: {text}")
    threading.Thread(target=_speak_thread_target, args=(text,), daemon=True).start()

voice_queue = None
main_loop = None

def stt_wake_word_callback(text: str):
    global is_speaking, expecting_response, response_timer
    if is_speaking: return 

    text_lower = text.lower()
    is_wake_word = "kai" in text_lower or "scooby" in text_lower
    
    if is_wake_word or expecting_response:
        expecting_response = False
        if response_timer:
            response_timer.cancel()
            
        if main_loop and main_loop.is_running() and voice_queue:
            main_loop.call_soon_threadsafe(voice_queue.put_nowait, text)

# --- ZMQ SETUP & TOOLS ---
zmq_ctx = zmq.asyncio.Context()

async def command_nav_controller(mode: str, vacuum) -> str:
    """Sends command to local Tier 2 with strict timeout handling."""
    socket = zmq_ctx.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5570")
    
    try:
        payload = {"mode": mode}
        if vacuum is not None:
            payload["vacuum"] = vacuum
            
        await socket.send_json(payload)
        res = await asyncio.wait_for(socket.recv_json(), timeout=2.0)
        return f"Confirmed. Status: {res.get('status', 'unknown')}"
    except asyncio.TimeoutError:
        return "Failed. Nav Engine (Tier 2) is offline or timed out."
    except Exception as e:
        return f"Failed. ZMQ Error - {e}"
    finally:
        socket.close()

async def trigger_camera() -> str:
    socket = zmq_ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{PI4_IP}:5561")
    
    try:
        await socket.send_json({"cmd": "camera"})
        res = await asyncio.wait_for(socket.recv_json(), timeout=5.0)
        
        if res.get("person_detected"):
            detections = res.get("detections", [])
            stance = detections[0].get("stance", "unknown") if detections else "unknown"
            return f"Camera: User detected. Current stance appears to be: {stance}."
        elif res.get("status") == "error":
            return f"Camera Error: {res.get('msg')}"
        return "Camera: No person detected in the room."
    except asyncio.TimeoutError:
        return "Camera Error: Timed out waiting for Pi 4."
    finally:
        socket.close()

# --- LANGGRAPH STATE ---
class BrainState(TypedDict):
    input_text: str
    context: Annotated[List[str], operator.add]
    agent_outputs: Annotated[List[str], operator.add]

llm = ChatOllama(model="llama3.1", temperature=0.2)

# --- WORKER NODES ---
async def analysis_agent(state: BrainState):
    global latest_heart_rate, ble_connected
    cam_result = await trigger_camera()
    hr_status = f"Current Heart Rate: {latest_heart_rate} BPM." if ble_connected else "Heart Rate Monitor is disconnected."
    combined_analysis = f"{cam_result} | {hr_status}"
    return {"context": [combined_analysis], "agent_outputs": [f"Analyst Observation: {combined_analysis}"]}

async def navigation_agent(state: BrainState):
    sys_msg = SystemMessage(content="You are the Navigation Router. Based on the user's prompt, determine the movement mode (forward, backward, left, right, stop, wander, home, or none) and vacuum state (true, false, or unchanged). Output ONLY exactly in this format: MODE: [mode], VAC: [true/false/unchanged].")
    res = await llm.ainvoke([sys_msg, HumanMessage(content=state["input_text"])])
    
    output = "No navigation needed."
    
    mode_match = re.search(r"MODE:\s*([a-zA-Z]+)", res.content, re.IGNORECASE)
    vac_match = re.search(r"VAC:\s*([a-zA-Z]+)", res.content, re.IGNORECASE)

    if mode_match:
        try:
            mode_str = mode_match.group(1).lower()
            vac_raw = vac_match.group(1).lower() if vac_match else "unchanged"
            vac_val = True if vac_raw == "true" else (False if vac_raw == "false" else None)
            
            nav_status = await command_nav_controller(mode_str, vac_val)
            output = f"HARDWARE ACTION TAKEN: Set mode to '{mode_str}' and vacuum to '{vac_raw}'. Hardware returned: {nav_status}"
        except Exception as e:
            output = f"HARDWARE ERROR: Failed to execute command. {e}"
        
    return {"context": [output], "agent_outputs": [f"Navigator says: {output}"]}

async def caregiver_agent(state: BrainState):
    global memory_summary, conversation_log, prompt_counter
    
    # FIX: Replaced negative constraints with a positive, natural persona.
    sys_msg = (
        "You are a helpful robotic assistant. Speak naturally and concisely in the first person ('I', 'me', 'my'). "
        "Rule: Never mention your own name in your responses. Just provide the answer or confirmation. "
        f"Long-term Memory: {memory_summary}\n"
    )
    
    recent_history = "\n".join(conversation_log[-6:]) if conversation_log else "No recent history."
    sys_msg += f"Recent Conversation Log:\n{recent_history}"
    
    if state.get("context"):
        # FIX: Tell the Caregiver to trust the system data and report it as its own actions.
        sys_msg += (
            "\n\n--- System Telemetry & Actions ---\n"
            "Your internal subsystems have executed the following actions in response to the user. "
            "Do NOT ask the user for clarification on these actions. Simply confirm to the user that you have done them.\n"
            f"Data: {state['context']}"
        )

    res = await llm.ainvoke([SystemMessage(content=sys_msg), HumanMessage(content=state["input_text"])])
    
    conversation_log.append(f"User: {state['input_text']}")
    conversation_log.append(f"Robot: {res.content}")
    prompt_counter += 1
    
    if prompt_counter >= 6:
        await trigger_memory_compaction()
        
    speak_text(res.content)
    
    return {"agent_outputs": [f"Caregiver said: {res.content}"]}

# --- SUPERVISOR NODE ---
def supervisor_router(state: BrainState):
    text = state["input_text"].lower()
    if any(word in text for word in ["look", "see", "health", "heart", "ok", "rate", "pulse"]):
        return "analyst"
    # Added "wander" to the trigger words so it always routes to the Navigator
    if any(word in text for word in ["stop", "come here", "go home", "clean", "vacuum", "move", "forward", "backward", "left", "right", "turn", "wander"]):
        return "navigator"
    return "caregiver"

# --- GRAPH ASSEMBLY ---
builder = StateGraph(BrainState)
builder.add_node("caregiver", caregiver_agent)
builder.add_node("navigator", navigation_agent)
builder.add_node("analyst", analysis_agent)

builder.add_conditional_edges(START, supervisor_router, ["caregiver", "navigator", "analyst"])
builder.add_edge("analyst", "caregiver")
builder.add_edge("navigator", "caregiver")
builder.add_edge("caregiver", END)

master_brain = builder.compile()

# --- ENTRY POINT ---
async def process_voice_commands():
    print("[BRAIN] Systems Online. Waiting for voice commands...")
    while True:
        user_input = await voice_queue.get()
        print(f"\n[WAKE WORD DETECTED] Processing: '{user_input}'\n")
        
        initial_state = {
            "input_text": user_input,
            "context": [],
            "agent_outputs": []
        }
        
        result = await master_brain.ainvoke(initial_state)
        
        print("\n--- Execution Summary ---")
        for out in result["agent_outputs"]:
            print(out)
        
        voice_queue.task_done()

async def main():
    print("Starting Tier 1 LangGraph Master Brain...")
    
    global main_loop, voice_queue
    main_loop = asyncio.get_running_loop()
    voice_queue = asyncio.Queue()
    
    print("[AUDIO] Initializing microphone and RealtimeSTT...")
    stt_thread = threading.Thread(
        target=start_stt_engine, 
        kwargs={
            "model": "base.en", 
            "wake_word_callback": stt_wake_word_callback, 
            "device": 1 # ReSpeaker ID
        },
        daemon=True
    )
    stt_thread.start()
    
    # Start background tasks
    asyncio.create_task(ble_heart_rate_loop())
    
    await process_voice_commands()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
