#!/usr/bin/env python3
"""
Tier 2 - Navigation & Math Engine (Runs on Laptop)
Receives high-level intents from LangGraph Master Brain.
Calculates Bug Algorithm and targets, sends exact wheel speeds to Pi 4.
"""

import time
import math
import zmq
import threading

# TODO: REPLACE WITH YOUR PI 4's ACTUAL IP ADDRESS
PI4_IP = "10.26.191.91" 

class NavigationController:
    def __init__(self):
        self.context = zmq.Context()
        
        # Telemetry SUB (From Pi 4)
        self.telem_sub = self.context.socket(zmq.SUB)
        self.telem_sub.connect(f"tcp://{PI4_IP}:5560")
        self.telem_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Command REQ (To Pi 4)
        self._init_req_socket()

        # Command REP (From Tier 1 Master Brain)
        self.brain_rep = self.context.socket(zmq.REP)
        self.brain_rep.bind("tcp://127.0.0.1:5570")

        # Physical Constants & State
        self.track_width = 0.33
        self.current_state = 1 # 0:Wander, 1:Stop, 2:Home, 3:Bug Bypass
        self.odom = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.lidar = []
        self.reflex_active = False
        
        # Bug Algorithm specific
        self.bypass_ticks = 0
        self.bypass_turn_dir = 1.0 

    def _init_req_socket(self):
        """Creates a socket with a timeout to prevent infinite blocking."""
        self.cmd_req = self.context.socket(zmq.REQ)
        self.cmd_req.setsockopt(zmq.RCVTIMEO, 1500) # 1.5s timeout crucial for self-healing
        self.cmd_req.connect(f"tcp://{PI4_IP}:5561")

    def send_to_pi(self, payload):
        """Sends data to Pi 4 and safely resets ZMQ state if Pi is unresponsive."""
        try:
            self.cmd_req.send_json(payload)
            return self.cmd_req.recv_json()
        except zmq.Again:
            print("[NAV] Warning: Pi 4 timed out. Destroying and recreating ZMQ socket.")
            # ZMQ REQ state machine is broken after a timeout. Must completely rebuild the socket.
            self.cmd_req.close(linger=0)
            self._init_req_socket()
            return {"status": "error", "msg": "Pi 4 timeout"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    def listen_to_brain(self):
        """Listens for high-level commands from the LangGraph Agent locally."""
        while True:
            msg = self.brain_rep.recv_json()
            mode = msg.get("mode", "none").lower()
            
            pi_res = {}
            # Decouple vacuum execution so it works even if mode is "none"
            if "vacuum" in msg:
                vac = msg["vacuum"]
                pi_res = self.send_to_pi({"cmd": "vacuum", "state": vac})
                print(f"[NAV] Vacuum set to {vac}. Pi Status: {pi_res.get('status')}")

            valid_modes = {
                "wander": 0, "stop": 1, "home": 2, 
                "forward": 4, "backward": 5, "left": 6, "right": 7
            }

            if mode in valid_modes:
                self.current_state = valid_modes[mode]
                print(f"[NAV] Brain requested {mode.upper()}.")
                self.brain_rep.send_json({"status": "success", "mode": mode, "pi_status": pi_res.get("status", "unchanged")})
            elif mode == "none":
                # User only asked to change the vacuum, keep current movement state
                self.brain_rep.send_json({"status": "success", "mode": "unchanged", "pi_status": pi_res.get("status", "unchanged")})
            else:
                self.brain_rep.send_json({"status": "error", "reason": "unknown mode"})

    def get_telemetry(self):
        """Continuously updates local map/state from Pi 4."""
        while True:
            try:
                data = self.telem_sub.recv_json(flags=zmq.NOBLOCK)
                self.odom = data["odom"]
                self.lidar = data["lidar"]
                self.reflex_active = data["reflex_active"]
            except zmq.Again:
                pass
            time.sleep(0.01)

    def convert_twist_to_wheels(self, linear, angular):
        left_speed = linear - (angular * self.track_width / 2.0)
        right_speed = linear + (angular * self.track_width / 2.0)
        self.send_to_pi({"cmd": "motor_speeds", "left": left_speed, "right": right_speed})

    def control_loop(self):
        """The core math engine. Replaces legacy ROS 2 logic."""
        while True:
            time.sleep(0.1) # 10Hz control loop
            
            if self.reflex_active:
                # Silently wait. Pi 4 has taken manual physical control.
                continue
                
            linear, angular = 0.0, 0.0

            if self.current_state == 0: # WANDER
                linear, angular = 0.2, 0.0
                
            elif self.current_state == 1: # STOP
                linear, angular = 0.0, 0.0
                
            elif self.current_state == 2: # HOME
                dist_to_home = math.sqrt(self.odom["x"]**2 + self.odom["y"]**2)
                angle_to_home = math.atan2(-self.odom["y"], -self.odom["x"])
                
                if dist_to_home > 0.1:
                    raw_error = angle_to_home - self.odom["theta"]
                    angle_error = math.atan2(math.sin(raw_error), math.cos(raw_error))
                    angular = angle_error * 1.5
                    linear = 0.15
                    
                    # Check LiDAR for obstacles blocking home (Bug Algorithm Trigger)
                    if len(self.lidar) == 360:
                        front_dist = min([self.lidar[i] for i in range(345, 360)] + [self.lidar[i] for i in range(0, 15)])
                        if 0.15 < front_dist < 0.5:
                            self.current_state = 3
                            self.bypass_ticks = 0
                            left_clear = self.lidar[90] if self.lidar[90] > 0 else 0
                            right_clear = self.lidar[270] if self.lidar[270] > 0 else 0
                            self.bypass_turn_dir = 1.0 if left_clear >= right_clear else -1.0
                            print("[NAV] Obstacle! Triggering Bug Algorithm.")
                else:
                    self.current_state = 0 # Reached origin
                    print("[NAV] Arrived home.")
            
            elif self.current_state == 3: # BUG ALGORITHM
                self.bypass_ticks += 1
                if self.bypass_ticks < 15: # Phase 1: Turn
                    linear, angular = 0.0, 1.0 * self.bypass_turn_dir
                elif self.bypass_ticks < 45: # Phase 2: Drive Straight
                    linear, angular = 0.2, 0.0
                else: # Phase 3: Resume
                    self.current_state = 2
            
            elif self.current_state == 4: # FORWARD
                linear, angular = 0.2, 0.0
                
            elif self.current_state == 5: # BACKWARD
                linear, angular = -0.2, 0.0
                
            elif self.current_state == 6: # TURN LEFT
                linear, angular = 0.0, 0.5
                
            elif self.current_state == 7: # TURN RIGHT
                linear, angular = 0.0, -0.5
            
            # Send computed commands down to the Pi
            self.convert_twist_to_wheels(linear, angular)

if __name__ == "__main__":
    print("Starting Tier 2 Navigation Engine...")
    nav = NavigationController()
    threading.Thread(target=nav.listen_to_brain, daemon=True).start()
    threading.Thread(target=nav.get_telemetry, daemon=True).start()
    nav.control_loop()
