#!/usr/bin/env python3
"""
Tier 3 - Edge Hardware Portal (Runs on Raspberry Pi 4)
Handles direct physical hardware, odometry, LiDAR reflex, BLE Heart Rate,
and manages a warm-booted IMX500 AI Camera.
Communicates with the Laptop (Tiers 1 & 2) via ZMQ.
"""

import time
import math
import threading
import zmq
import cv2
import json
from gpiozero import PWMOutputDevice, DigitalOutputDevice, RotaryEncoder
from rplidar import RPLidar, RPLidarException

# --- HARDWARE CONFIGURATION ---
WHEEL_RADIUS = 0.045
TRACK_WIDTH = 0.27  # 27cm track width
TICKS_PER_REV = 360.0

# 50% Speed - Snappy but safe for demo power limits
REFLEX_TURN_SPEED = 0.50  
REFLEX_DRIVE_SPEED = 0.50 

vacuum = DigitalOutputDevice(26)
vacuum.off()

motor_left_pwm = PWMOutputDevice(12)
motor_left_dir = DigitalOutputDevice(13)
motor_right_pwm = PWMOutputDevice(18)
motor_right_dir = DigitalOutputDevice(19)

encoder_left = RotaryEncoder(17, 27, max_steps=0)
encoder_right = RotaryEncoder(22, 23, max_steps=0)

# --- GLOBAL STATE ---
odom_state = {"x": 0.0, "y": 0.0, "theta": 0.0}
lidar_scan = []  
reflex_active = False

# --- WARM BOOT THE CAMERA ---
print("[SYSTEM] Initializing IMX500 AI Camera (May take up to 2 mins)...")
try:
    from camera_detect import CameraDetector
    camera_detector = CameraDetector(
        model_path="imx500-models/imx500_network_yolo11n_pp.rpk",
        labels_path="imx500-models/coco_labels.txt"
    )
except Exception as e:
    print(f"[WARN] Camera failed to boot: {e}")
    camera_detector = None

def set_motor(pwm_dev, dir_dev, speed, invert=False):
    """Sets motor speed and direction. Invert flag fixes reversed wiring."""
    if invert:
        speed = -speed
        
    speed = max(-1.0, min(1.0, speed))
    if speed >= 0:
        dir_dev.off()
        pwm_dev.value = speed
    else:
        dir_dev.on()
        pwm_dev.value = abs(speed)

def odometry_loop():
    global odom_state
    last_left = 0
    last_right = 0
    meters_per_tick = (2.0 * math.pi * WHEEL_RADIUS) / TICKS_PER_REV

    while True:
        curr_left, curr_right = encoder_left.steps, encoder_right.steps
        d_left = (curr_left - last_left) * meters_per_tick
        d_right = (curr_right - last_right) * meters_per_tick
        last_left, last_right = curr_left, curr_right

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / TRACK_WIDTH

        odom_state["x"] += d_center * math.cos(odom_state["theta"] + d_theta / 2.0)
        odom_state["y"] += d_center * math.sin(odom_state["theta"] + d_theta / 2.0)
        odom_state["theta"] += d_theta
        
        time.sleep(0.02) # 50Hz

# --- 3. LIDAR & 360-DEGREE ESCAPE REFLEX ---
def lidar_and_reflex_loop():
    global lidar_scan, reflex_active
    lidar = None
    
    while True:
        try:
            print("[LIDAR] Connecting to /dev/ttyUSB0...")
            lidar = RPLidar('/dev/ttyUSB0')
            print("[LIDAR] Connected successfully. Starting scan...")
            
            for scan in lidar.iter_scans():
                current_scan = [10.0] * 360 
                for (_, angle, distance) in scan:
                    dist_m = distance / 1000.0
                    if dist_m > 0:
                        # Shift physical 180 to software 0
                        shifted_angle = (int(angle) + 180) % 360
                        current_scan[shifted_angle] = dist_m
                lidar_scan = current_scan

                # --- UPGRADED WIDE-ANGLE SAFETY ZONES ---
                # Front: 340 to 20 (40 degree wide cone)
                front_cone = current_scan[-20:] + current_scan[:20]
                # Right: 60 to 120 (60 degree wide side-shield)
                right_cone = current_scan[60:120]
                # Rear: 140 to 220 (80 degree anti-chair wide sweep)
                rear_cone = current_scan[140:220]
                # Left: 240 to 300 (60 degree wide side-shield)
                left_cone = current_scan[240:300]

                # Filter invalid points
                valid_front = [r for r in front_cone if 0.05 < r < 10.0]
                valid_right = [r for r in right_cone if 0.05 < r < 10.0]
                valid_rear = [r for r in rear_cone if 0.05 < r < 10.0]
                valid_left = [r for r in left_cone if 0.05 < r < 10.0]
                
                # Get minimum distances
                dist_front = min(valid_front) if valid_front else 10.0
                dist_right = min(valid_right) if valid_right else 10.0
                dist_rear = min(valid_rear) if valid_rear else 10.0
                dist_left = min(valid_left) if valid_left else 10.0
                
                # --- EVALUATE PRIORITY HIERARCHY ---
                
                # Priority 1: Front Crash Imminent (< 30cm)
                if dist_front < 0.30:
                    reflex_active = True
                    # Evaluate Front-Left vs Front-Right to choose spin direction
                    front_left_view = [r for r in current_scan[315:360] if 0.05 < r < 10.0]
                    front_right_view = [r for r in current_scan[0:45] if 0.05 < r < 10.0]
                    
                    dl = min(front_left_view) if front_left_view else 10.0
                    dr = min(front_right_view) if front_right_view else 10.0
                    
                    if dr < dl:
                        print("[REFLEX] Blocked Front-Right! Spinning Left.")
                        set_motor(motor_left_pwm, motor_left_dir, -REFLEX_TURN_SPEED, invert=True)
                        set_motor(motor_right_pwm, motor_right_dir, REFLEX_TURN_SPEED)
                    else:
                        print("[REFLEX] Blocked Front-Left/Center! Spinning Right.")
                        set_motor(motor_left_pwm, motor_left_dir, REFLEX_TURN_SPEED, invert=True)
                        set_motor(motor_right_pwm, motor_right_dir, -REFLEX_TURN_SPEED)

                # Priority 2: Left Side Crash Imminent (< 25cm bubble)
                elif dist_left < 0.25:
                    reflex_active = True
                    print("[REFLEX] Threat detected directly LEFT! Spinning Right.")
                    set_motor(motor_left_pwm, motor_left_dir, REFLEX_TURN_SPEED, invert=True)
                    set_motor(motor_right_pwm, motor_right_dir, -REFLEX_TURN_SPEED)

                # Priority 3: Right Side Crash Imminent (< 25cm bubble)
                elif dist_right < 0.25:
                    reflex_active = True
                    print("[REFLEX] Threat detected directly RIGHT! Spinning Left.")
                    set_motor(motor_left_pwm, motor_left_dir, -REFLEX_TURN_SPEED, invert=True)
                    set_motor(motor_right_pwm, motor_right_dir, REFLEX_TURN_SPEED)

                # Priority 4: Object detected in Rear Sweep (< 45cm)
                elif dist_rear < 0.45:
                    reflex_active = True
                    print("[REFLEX] Object approaching REAR! Driving Forward.")
                    set_motor(motor_left_pwm, motor_left_dir, REFLEX_DRIVE_SPEED, invert=True)
                    set_motor(motor_right_pwm, motor_right_dir, REFLEX_DRIVE_SPEED)
                
                # Path is Clear
                else:
                    if reflex_active:
                        print("[REFLEX] All perimeters clear. Returning control to ZMQ.")
                        set_motor(motor_left_pwm, motor_left_dir, 0.0, invert=True)
                        set_motor(motor_right_pwm, motor_right_dir, 0.0)
                        reflex_active = False
                    
        except Exception as e:
            print(f"[WARN] LiDAR exception: {e}. Attempting to reset...")
            if lidar:
                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except:
                    pass
            time.sleep(3) 

# --- ZMQ COMMS LOOP ---
def zmq_comms_loop():
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5560")
    
    rep_socket = context.socket(zmq.REP)
    rep_socket.bind("tcp://*:5561")

    print("[ZMQ] Tier 3 Pi 4 Portals Open. Awaiting connections...")

    while True:
        try:
            msg = rep_socket.recv_json(flags=zmq.NOBLOCK)
            cmd = msg.get("cmd")
            
            # ZMQ commands ONLY work if the reflex is NOT overriding the motors
            if cmd == "motor_speeds" and not reflex_active:
                set_motor(motor_left_pwm, motor_left_dir, msg.get("left", 0.0), invert=True)
                set_motor(motor_right_pwm, motor_right_dir, msg.get("right", 0.0))
                rep_socket.send_json({"status": "ok"})
                
            elif cmd == "vacuum":
                vacuum.on() if msg.get("state") else vacuum.off()
                rep_socket.send_json({"status": "ok"})
                
            elif cmd == "camera":
                if camera_detector:
                    snap = camera_detector.get_snapshot()
                    rep_socket.send_json({
                        "status": "success",
                        "person_detected": len(snap.get("detections", [])) > 0,
                        "detections": snap.get("detections", [])
                    })
                else:
                    rep_socket.send_json({"status": "error", "msg": "Camera offline"})
                    
            else:
                rep_socket.send_json({"status": "ignored_due_to_reflex" if reflex_active else "unknown_cmd"})
        except zmq.Again:
            pass

        # Publish Telemetry
        telemetry = {
            "odom": odom_state,
            "lidar": lidar_scan,
            "reflex_active": reflex_active
        }
        pub_socket.send_json(telemetry)
        time.sleep(0.05) 

if __name__ == "__main__":
    print("Starting Edge Hardware Portal (Pi 4)...")
    threading.Thread(target=odometry_loop, daemon=True).start()
    threading.Thread(target=lidar_and_reflex_loop, daemon=True).start()
    
    try:
        zmq_comms_loop()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
    finally:
        if camera_detector:
            camera_detector.close()
            print("[CAM] Camera released.")
