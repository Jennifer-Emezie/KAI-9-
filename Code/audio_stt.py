#!/usr/bin/env python3
"""
Real-time Speech-to-Text using ReSpeaker USB Mic Array v2.0 + RealtimeSTT
Integrated with Kai 9's Pi 4 Sensory Edge Architecture.
"""

import argparse
import sys
import os
import shutil
from datetime import datetime

# ── RTL display helper ────────────────────────────────────────────────────────

RTL_RANGES = [
    ('\u0600', '\u06FF'),  # Arabic
    ('\u0590', '\u05FF'),  # Hebrew
    ('\u0750', '\u077F'),  # Arabic Supplement
    ('\uFB50', '\uFDFF'),  # Arabic Presentation Forms-A
    ('\uFE70', '\uFEFF'),  # Arabic Presentation Forms-B
]

def is_rtl(text: str) -> bool:
    return any(lo <= c <= hi for c in text for lo, hi in RTL_RANGES)

def bidi(text: str) -> str:
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except ImportError:
        return text

# ── Windows terminal: enable ANSI escape codes ────────────────────────────────

if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ── Output file helper ────────────────────────────────────────────────────────

GLOBAL_OUTPUT_FILE = None
WAKE_WORD_CALLBACK = None  # 🚀 NEW: We store the callback here dynamically

def save(text: str):
    if GLOBAL_OUTPUT_FILE:
        with open(GLOBAL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")

# ── Callbacks ─────────────────────────────────────────────────────────────────

def on_partial(text: str):
    if not text:
        return
    out = bidi(text) if is_rtl(text) else text
    
    # 🚀 NEW: Truncate wide text to prevent terminal wrapping spasming
    term_width = shutil.get_terminal_size().columns - 5
    if len(out) > term_width:
        out = "..." + out[-(term_width - 3):]
        
    print(f"\r\033[K  {out}", end="", flush=True)

def on_final(text: str):
    if not text.strip():
        return
    out = bidi(text) if is_rtl(text) else text
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if is_rtl(text):
        print(f"\r\033[K{out}  [{timestamp}]", flush=True)
    else:
        print(f"\r\033[K[{timestamp}] {out}", flush=True)
        
    save(text)

    # 🚀 THE MAGIC LINK: Call the injected Wake Word router (if it exists)
    if WAKE_WORD_CALLBACK:
        WAKE_WORD_CALLBACK(text)

# ── Core Engine (Encapsulated for Threading) ──────────────────────────────────

def start_stt_engine(model="base.en", language=None, device=None, output=None, wake_word_callback=None, **kwargs):
    """
    Main entry point for RealtimeSTT. 
    Can be called directly from pi4_edge_node.py inside a daemon thread.
    Accepts **kwargs to allow passing dynamic_energy_threshold, energy_threshold, etc.
    """
    global GLOBAL_OUTPUT_FILE, WAKE_WORD_CALLBACK
    GLOBAL_OUTPUT_FILE = output
    WAKE_WORD_CALLBACK = wake_word_callback
    
    from RealtimeSTT import AudioToTextRecorder

    print(f"Loading Whisper '{model}' model …", flush=True)

    recorder_config = {
        "model": model,
        "language": language,
        "input_device_index": device,

        # Live partial transcription (OPTIMIZED FOR PI 4)
        "enable_realtime_transcription": True,
        "realtime_processing_pause": 0.2,   # 🚀 Relaxed loop to free up CPU cores
        "realtime_model_type": "tiny.en",   # 🚀 Use the fastest possible model for live feedback
        "on_realtime_transcription_update": on_partial,

        # Final transcription parameters
        "silero_sensitivity": 0.4,
        "webrtc_sensitivity": 2,
        "post_speech_silence_duration": 0.4,
        "min_length_of_recording": 0.5,
        "min_gap_between_recordings": 0.1,

        "spinner": False,
    }

    # Inject the loud room presentation settings (and any other kwargs passed from Tier 1)
    recorder_config.update(kwargs)

    recorder = AudioToTextRecorder(**recorder_config)

    print("✓ Model ready.\n")
    if device is not None:
        print(f"  Using device [{device}]")
    else:
        print("  Using system default input device")
    print("\n🎙  Listening … say 'Kai' or 'Scooby' to issue a command. Ctrl+C to stop.\n")

    try:
        while True:
            recorder.text(on_final)
    except KeyboardInterrupt:
        print("\n\n── Session ended ──")
        if output:
            print(f"Transcript saved to: {output}")
    finally:
        try:
            recorder.stop()
        except:
            pass

# ── Standalone CLI Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReSpeaker → Real-time Whisper STT")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--model", default="base.en",
                        choices=["tiny", "tiny.en", "base", "base.en", "small"],
                        help="Whisper model size (default: base.en)")
    parser.add_argument("--output", default=None, help="File to append final transcriptions to")
    parser.add_argument("--language", default=None, help="Force language code e.g. 'en', 'ar'")
    
    # Optional: Allow testing the energy threshold from CLI too
    parser.add_argument("--energy-threshold", type=int, default=None, help="Override energy threshold for loud rooms")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\nAvailable audio INPUT devices:")
        print(f"  {'#':<4} {'Max ch':<8} Name")
        print("  " + "-" * 50)
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                print(f"  {i:<4} {int(d['max_input_channels']):<8} {d['name']}")
        print()
        sys.exit(0)

    if args.device is None:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if "respeaker" in d["name"].lower() or "seeed" in d["name"].lower():
                if d["max_input_channels"] > 0:
                    args.device = i
                    break

    kwargs = {}
    if args.energy_threshold is not None:
        kwargs["energy_threshold"] = args.energy_threshold
        kwargs["dynamic_energy_threshold"] = False

    start_stt_engine(
        model=args.model,
        language=args.language,
        device=args.device,
        output=args.output,
        **kwargs
    )
