import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import socket
import threading
import json
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import math  # One Euro filter needs math
import os

# -------- One Euro Filter (low-latency smoothing) --------
class _LowPassFilter:
    def __init__(self, alpha, initval=None):
        self.alpha = alpha
        self.s = initval
        self.initialized = initval is not None

    def apply(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if not self.initialized:
            self.s = x
            self.initialized = True
            return x
        self.s = self.alpha * x + (1.0 - self.alpha) * self.s
        return self.s

def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.3, d_cutoff=1.0, initval=None, init_time=None):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = _LowPassFilter(alpha=1.0, initval=initval)
        self.dx_filter = _LowPassFilter(alpha=1.0, initval=0.0 if initval is not None else None)
        self.last_time = init_time

    def apply(self, x, t=None):
        now = t if t is not None else time.time()
        if self.last_time is None:
            dt = 1.0 / 60.0  # assume ~60 FPS on first call
        else:
            dt = max(1e-3, now - self.last_time)
        self.last_time = now

        # Estimate derivative
        dx = 0.0
        if self.x_filter.initialized:
            dx = (x - self.x_filter.s) / dt
        edx = self.dx_filter.apply(dx, _alpha(self.d_cutoff, dt))

        # Dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        # Filtered signal
        return self.x_filter.apply(x, _alpha(cutoff, dt))

class Ball:
    def __init__(self, x, y, radius=30):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.gravity = 0.5
        self.damping = 0.98
        self.bounce_damping = 0.7
        self.max_speed = 50  # Maximum speed in pixels per frame
    
    def update(self, width, height, floor_height=100):
        # Apply gravity
        self.vy += self.gravity
        
        # Apply damping
        self.vx *= self.damping
        self.vy *= self.damping
        
        # Cap velocity to max speed
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Wall collisions (both sides)
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * self.bounce_damping
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx = -self.vx * self.bounce_damping
        
        # Top wall
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy * self.bounce_damping
        
        # Floor collision (raised floor)
        floor_y = height - floor_height
        if self.y + self.radius > floor_y:
            self.y = floor_y - self.radius
            self.vy = -self.vy * self.bounce_damping
            # Add friction when on ground
            self.vx *= 0.95
    
    def apply_force(self, fx, fy):
        self.vx += fx
        self.vy += fy
    
    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (0, 255, 255), -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (0, 200, 200), 2)
    
    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy
        }
    
    def from_dict(self, data):
        self.x = data['x']
        self.y = data['y']
        self.vx = data['vx']
        self.vy = data['vy']

class Hand:
    def __init__(self, radius=50):
        self.x = None
        self.y = None
        # Target position (from detection/network)
        self.target_x = None
        self.target_y = None
        # Smoothed/drawn position (interpolated)
        self.display_x = None
        self.display_y = None
        self.radius = radius
        self.is_active = False
        self.last_seen_time = 0
        self.timeout = 0.5  # seconds before hand disappears
        
        # Linear interpolation settings for smooth catch-up
        self.lerp_speed = 0.35  # Smoothness factor (0-1), higher = faster catch-up
        self.max_lerp_distance = 300  # Max pixels per frame to catch up (prevents extreme jumps)

        # One Euro filters for low-latency smoothing on the target position
        # Tune min_cutoff/beta to trade jitter vs. responsiveness
        self._filter_x = OneEuroFilter(min_cutoff=1.2, beta=0.3, d_cutoff=1.0)
        self._filter_y = OneEuroFilter(min_cutoff=1.2, beta=0.3, d_cutoff=1.0)

    def update(self, center_x, center_y, t=None):
        """Update hand position when detected; smooth to avoid teleportation."""
        now = t if t is not None else time.time()
        was_active = self.is_active

        self.x = center_x
        self.y = center_y
        self.is_active = True
        self.last_seen_time = now

        # If this is the first-ever sample, or we have no display pos yet, snap once.
        if self.display_x is None or self.display_y is None:
            self.target_x = center_x
            self.target_y = center_y
            self.display_x = center_x
            self.display_y = center_y
            # Re-init filters from this starting point
            self._filter_x = OneEuroFilter(min_cutoff=1.2, beta=0.3, d_cutoff=1.0, initval=self.target_x, init_time=now)
            self._filter_y = OneEuroFilter(min_cutoff=1.2, beta=0.3, d_cutoff=1.0, initval=self.target_y, init_time=now)
            return

        # If we just re-acquired the hand after a timeout, start smoothing from last drawn pos
        if not was_active:
            self._filter_x = OneEuroFilter(min_cutoff=1.2, beta=0.3, d_cutoff=1.0, initval=self.display_x, init_time=now)
            self._filter_y = OneEuroFilter(min_cutoff=1.2, beta=0.3, d_cutoff=1.0, initval=self.display_y, init_time=now)

        # Apply filter to get target position (filtered version of raw input)
        self.target_x = self._filter_x.apply(center_x, now)
        self.target_y = self._filter_y.apply(center_y, now)

    def update_display_position(self):
        """Linearly interpolate display position towards target to avoid teleporting."""
        if self.target_x is None or self.target_y is None:
            return
        
        if self.display_x is None or self.display_y is None:
            self.display_x = self.target_x
            self.display_y = self.target_y
            return
        
        # Calculate distance to target
        dx = self.target_x - self.display_x
        dy = self.target_y - self.display_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.5:  # Close enough, snap to target
            self.display_x = self.target_x
            self.display_y = self.target_y
        else:
            # Linear interpolation towards target
            # Use lerp_speed but cap the maximum distance per frame
            lerp_distance = min(distance * self.lerp_speed, self.max_lerp_distance)
            
            # Move display position towards target
            if distance > 0:
                move_ratio = lerp_distance / distance
                self.display_x += dx * move_ratio
                self.display_y += dy * move_ratio

    def check_timeout(self):
        """Check if hand should disappear due to timeout"""
        if self.is_active and (time.time() - self.last_seen_time) > self.timeout:
            self.is_active = False
            # Keep display_x/display_y so we can smoothly continue when re-appearing

    def draw(self, frame):
        """Draw the hand circle (smoothed position)"""
        if self.is_active and self.display_x is not None and self.display_y is not None:
            overlay = frame.copy()
            cv2.circle(overlay, (int(self.display_x), int(self.display_y)), self.radius, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.circle(frame, (int(self.display_x), int(self.display_y)), self.radius, (0, 255, 0), 3)
            cv2.circle(frame, (int(self.display_x), int(self.display_y)), 5, (255, 0, 0), -1)

    def check_collision(self, ball):
        """Check if hand collides with ball (use smoothed/drawn position)"""
        if not self.is_active or self.display_x is None or self.display_y is None:
            return False
        dx = self.display_x - ball.x
        dy = self.display_y - ball.y
        distance = np.sqrt(dx**2 + dy**2)
        return distance < (self.radius + ball.radius)

    def to_dict(self):
        # Share the target position over the network (the filtered input position)
        return {
            'x': self.target_x,
            'y': self.target_y,
            'is_active': self.is_active
        }

    def from_dict(self, data):
        # For peer hands, set the target position and let it interpolate
        if data:
            new_x = data.get('x')
            new_y = data.get('y')
            self.is_active = data.get('is_active', False)
            
            if new_x is not None and new_y is not None:
                # Initialize if first time
                if self.target_x is None or self.target_y is None:
                    self.target_x = new_x
                    self.target_y = new_y
                    self.display_x = new_x
                    self.display_y = new_y
                else:
                    # Update target position (will be interpolated in update_display_position)
                    self.target_x = new_x
                    self.target_y = new_y
                
                self.x = self.target_x
                self.y = self.target_y
                
            if self.is_active:
                self.last_seen_time = time.time()

class NetworkManager:
    def __init__(self, port=5555):
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.connection = None
        self.client_address = None
        
        self.is_server = False
        self.connected = False
        self.is_left_player = True
        
        self.received_data = None
        self.last_receive_time = 0
        self.authority_transfer_pending = False
        
        self.running = True
        self.receive_thread = None
    
    def start_server(self):
        """Start TCP server and wait for connection"""
        self.is_server = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)
        
        print(f"Server listening on port {self.port}...")
        
        # Start accept thread
        accept_thread = threading.Thread(target=self._accept_connection, daemon=True)
        accept_thread.start()
    
    def _accept_connection(self):
        """Accept incoming connection"""
        while self.running and not self.connected:
            try:
                conn, addr = self.server_socket.accept()
                print(f"Connection from {addr}")
                
                # Receive initial handshake
                data = self._recv_message(conn)
                if data and data['type'] == 'handshake':
                    # Ask user to confirm
                    root = tk.Tk()
                    root.withdraw()
                    result = messagebox.askyesno(
                        "Connection Request",
                        f"Player wants to connect from {addr[0]}\n"
                        f"They are: {data['position']}\n"
                        f"Accept?"
                    )
                    root.destroy()
                    
                    if result:
                        self.connection = conn
                        self.client_address = addr
                        self.is_left_player = (data['position'] == 'right')
                        
                        # Send acceptance
                        position = 'left' if self.is_left_player else 'right'
                        self._send_message({
                            'type': 'handshake_accept',
                            'position': position
                        }, conn)
                        
                        self.connected = True
                        print(f"Connection accepted! You are: {position}")
                        
                        # Start receive thread
                        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                        self.receive_thread.start()
                    else:
                        conn.close()
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")
    
    def connect_to_server(self, host):
        """Connect to TCP server as client"""
        self.is_server = False
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.settimeout(10.0)
        
        try:
            print(f"Connecting to {host}:{self.port}...")
            self.client_socket.connect((host, self.port))
            
            # Send handshake
            position = 'left' if self.is_left_player else 'right'
            self._send_message({
                'type': 'handshake',
                'position': position
            }, self.client_socket)
            
            # Wait for acceptance
            response = self._recv_message(self.client_socket)
            if response and response['type'] == 'handshake_accept':
                self.connection = self.client_socket
                self.connected = True
                print(f"Connected! Peer is: {response['position']}")
                
                # Start receive thread
                self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self.receive_thread.start()
                return True
            else:
                print("Connection rejected")
                self.client_socket.close()
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            if self.client_socket:
                self.client_socket.close()
            return False
    
    def _recv_message(self, sock):
        """Receive length-prefixed JSON message"""
        try:
            # First receive 4-byte length prefix
            length_data = b''
            while len(length_data) < 4:
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return None
                length_data += chunk
            
            message_length = int.from_bytes(length_data, 'big')
            
            # Receive the actual message
            message_data = b''
            while len(message_data) < message_length:
                chunk = sock.recv(message_length - len(message_data))
                if not chunk:
                    return None
                message_data += chunk
            
            return json.loads(message_data.decode('utf-8'))
        except Exception as e:
            print(f"Receive error: {e}")
            return None
    
    def _send_message(self, message, sock=None):
        """Send length-prefixed JSON message"""
        if sock is None:
            sock = self.connection
        
        if sock is None:
            return
        
        try:
            data = json.dumps(message).encode('utf-8')
            length = len(data).to_bytes(4, 'big')
            sock.sendall(length + data)
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
    
    def _receive_loop(self):
        """Main receive loop for handling messages"""
        while self.running and self.connected:
            try:
                message = self._recv_message(self.connection)
                if message is None:
                    print("Connection closed by peer")
                    self.connected = False
                    break
                
                if message['type'] == 'game_state':
                    self.received_data = message
                    self.last_receive_time = time.time()
                elif message['type'] == 'ping':
                    self._send_message({'type': 'pong'})
                    
            except Exception as e:
                if self.running:
                    print(f"Receive loop error: {e}")
                self.connected = False
                break
    
    def send_game_state(self, ball, hand, score, has_authority, is_host, authority_reset=False, authority_owner=None):
        """Send complete game state to peer"""
        if self.connected:
            self._send_message({
                'type': 'game_state',
                'ball': ball.to_dict(),
                'hand': hand.to_dict(),
                'score': score,
                'has_authority': has_authority,
                'is_host': is_host,
                'authority_reset': authority_reset,
                'authority_owner': authority_owner
            })
    
    def get_received_data(self):
        """Get received game state"""
        return self.received_data
    
    def close(self):
        """Close all connections"""
        self.running = False
        
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

def get_hand_polygon(hand_landmarks, width, height):
    points = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    return hull

def setup_connection():
    root = tk.Tk()
    root.withdraw()
    
    # Ask if host or join
    choice = messagebox.askquestion(
        "Multiplayer Setup",
        "Do you want to HOST a game?\n\n"
        "YES = Host (wait for connection)\n"
        "NO = Join (connect to IP)"
    )
    
    if choice == 'yes':
        # Host mode - wait for connection
        local_ip = socket.gethostbyname(socket.gethostname())
        messagebox.showinfo(
            "Host Mode",
            f"Your IP: {local_ip}\n\n"
            f"Share this with your friend.\n"
            f"Waiting for connection..."
        )
        
        side_choice = messagebox.askquestion("Choose Side", "Do you want to be on the LEFT?")
        is_left = (side_choice == 'yes')
        
        root.destroy()
        return None, is_left, True  # host mode
    else:
        # Join mode - enter IP
        peer_ip = simpledialog.askstring("Join Game", "Enter host IP address:")
        if not peer_ip:
            root.destroy()
            return None, True, False
        
        side_choice = messagebox.askquestion("Choose Side", "Do you want to be on the LEFT?")
        is_left = (side_choice == 'yes')
        
        root.destroy()
        return peer_ip, is_left, False  # join mode

def main():
    # Setup connection
    peer_ip, is_left_player, is_host = setup_connection()
    
    # Initialize network
    network = NetworkManager()
    network.is_left_player = is_left_player
    
    if is_host:
        network.start_server()
    else:
        if not peer_ip or not network.connect_to_server(peer_ip):
            print("Failed to connect to server")
            return
    
    # Initialize MediaPipe Hands (prefer GPU Tasks if available)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    use_tasks_gpu = False
    hand_task = None
    try:
        # Try MediaPipe Tasks Hand Landmarker with GPU
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if os.path.exists(model_path):
            options = HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path,
                    delegate=BaseOptions.Delegate.GPU  # try GPU
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            hand_task = HandLandmarker.create_from_options(options)
            use_tasks_gpu = True
            print("Using MediaPipe Tasks HandLandmarker with GPU delegate")
        else:
            print("hand_landmarker.task not found; using mp.solutions Hands (CPU)")
    except Exception as e:
        print(f"GPU Tasks init failed or not available: {e}\nFalling back to mp.solutions Hands (CPU)")

    # Fallback CPU solution (lighter model for speed)
    hands = None
    if not use_tasks_gpu:
        hands = mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=1,
            model_complexity=0  # speed-up on CPU
        )

    # Initialize webcam (lower resolution, higher FPS if possible)
    cap = cv2.VideoCapture(0)
    # Request MJPG to unlock higher FPS on many webcams
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    # Request higher FPS and lower resolution
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Reduce internal buffering to lower latency (may be ignored on some backends)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Read back actual capture properties
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display dimensions (double width for split screen)
    display_width = cam_width * 2
    display_height = cam_height
    
    # Floor settings
    floor_height = 100
    
    # Initialize ball (start on left side)
    ball = Ball(cam_width // 2, display_height // 2)
    
    # Initialize hands
    my_hand = Hand(radius=50)
    peer_hand = Hand(radius=50)
    
    score = {'left': 0, 'right': 0}
    hand_positions = deque(maxlen=5)

    # Authority ownership controlled by the host only.
    authority_owner = 'host'
    local_has_control = is_host  # initially host controls

    # Track previous ball position
    prev_ball_x, prev_ball_y = ball.x, ball.y

    # Host-side anomaly and authority management
    last_interaction_time = time.time()
    authority_timeout = 15.0  # seconds
    client_timeout = 0.8      # seconds without client updates -> revoke (increased)
    max_jump_per_frame = 250  # px jump in 1 frame -> revoke
    pos_margin = 100          # px allowed beyond bounds before revoke

    # Hysteresis to avoid flapping
    hysteresis_margin = 40        # px margin inside a half before switching
    switch_dwell_time = 0.25      # seconds the ball must dwell on a half before switching
    authority_cooldown = 0.50     # seconds after a switch before another switch allowed
    last_authority_change_time = time.time()
    # Track which side is currently "desired" based on hysteresis and since when
    desired_side = None           # 'host' | 'client' | None
    desired_since = time.time()

    # Track last time host received a client ball update (for timeout)
    last_client_ball_rx_time = time.time()

    def is_on_client_side(x):
        # Host perspective: which half belongs to the client
        if is_left_player:   # host is left
            return x >= cam_width
        else:                # host is right
            return x < cam_width

    def is_on_client_side_with_margin(x):
        # Apply hysteresis margin
        if is_left_player:   # host is left
            return x >= (cam_width + hysteresis_margin)
        else:
            return x <= (cam_width - hysteresis_margin)

    def is_on_host_side_with_margin(x):
        if is_left_player:   # host is left
            return x <= (cam_width - hysteresis_margin)
        else:
            return x >= (cam_width + hysteresis_margin)

    def is_in_center_band(x):
        return (cam_width - hysteresis_margin) < x < (cam_width + hysteresis_margin)

    def ball_state_is_valid(candidate, prev_x, prev_y):
        try:
            x = float(candidate['x']); y = float(candidate['y'])
            vx = float(candidate['vx']); vy = float(candidate['vy'])
        except Exception:
            return False
        if not np.isfinite([x, y, vx, vy]).all():
            return False
        if x < -pos_margin or x > (display_width + pos_margin):
            return False
        if y < -pos_margin or y > (display_height + pos_margin):
            return False
        if abs(x - prev_x) > max_jump_per_frame or abs(y - prev_y) > max_jump_per_frame:
            return False
        return True

    print("\n=== MULTIPLAYER HAND BALL GAME (SPLIT SCREEN) ===")
    print(f"You are: {'LEFT' if is_left_player else 'RIGHT'} player")
    print(f"Mode: {'SERVER (HOST)' if is_host else 'CLIENT'}")
    print("Hit the ball to the opponent's side!")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create double-width canvas
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Draw floor on full canvas
        floor_y = display_height - floor_height
        cv2.rectangle(canvas, (0, floor_y), (display_width, display_height), (139, 69, 19), -1)
        cv2.line(canvas, (0, floor_y), (display_width, floor_y), (100, 50, 0), 3)
        
        # Place camera feed on appropriate side
        if is_left_player:
            canvas[0:cam_height, 0:cam_width] = frame
        else:
            canvas[0:cam_height, cam_width:display_width] = frame
        
        # Draw center dividing line
        cv2.line(canvas, (cam_width, 0), (cam_width, display_height), (255, 255, 255), 4)
        cv2.putText(canvas, "LEFT", (cam_width - 80, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "RIGHT", (cam_width + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Process hand detection (GPU Tasks if available, otherwise CPU Solutions)
        if use_tasks_gpu and hand_task is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = hand_task.detect_for_video(mp_image, int(time.time() * 1000))
                if result and result.hand_landmarks:
                    for landmarks in result.hand_landmarks:
                        # Compute hand center from normalized landmarks
                        xs = [int(l.x * cam_width) for l in landmarks]
                        ys = [int(l.y * cam_height) for l in landmarks]
                        hand_center_x = int(sum(xs) / len(xs))
                        hand_center_y = int(sum(ys) / len(ys))
                        # Offset to full canvas position
                        canvas_x = hand_center_x if is_left_player else (cam_width + hand_center_x)
                        my_hand.update(canvas_x, hand_center_y)
                        if my_hand.display_x is not None and my_hand.display_y is not None:
                            hand_positions.append((my_hand.display_x, my_hand.display_y))
            except Exception as e:
                # If anything goes wrong, skip this frame's detection
                # (fallback is already selected at init time)
                pass
        else:
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_polygon = get_hand_polygon(hand_landmarks, cam_width, cam_height)
                    M = cv2.moments(hand_polygon)
                    if M["m00"] != 0:
                        hand_center_x = int(M["m10"] / M["m00"])
                        hand_center_y = int(M["m01"] / M["m00"])
                        canvas_x = hand_center_x if is_left_player else (cam_width + hand_center_x)
                        my_hand.update(canvas_x, hand_center_y)
                        if my_hand.display_x is not None and my_hand.display_y is not None:
                            hand_positions.append((my_hand.display_x, my_hand.display_y))

        # Check hand timeout
        my_hand.check_timeout()
        
        # Update hand display positions (smooth interpolation towards target)
        my_hand.update_display_position()
        peer_hand.update_display_position()
        
        # Calculate hand velocity (based on smoothed positions)
        hand_vx, hand_vy = 0, 0
        if len(hand_positions) >= 2:
            hand_vx = (hand_positions[-1][0] - hand_positions[0][0]) / len(hand_positions)
            hand_vy = (hand_positions[-1][1] - hand_positions[0][1]) / len(hand_positions)
        
        # To send this frame (host may set on anomaly)
        authority_reset_to_send = False

        # Get peer data
        peer_data = network.get_received_data()
        if peer_data:
            # Always update peer hand
            if 'hand' in peer_data:
                peer_hand.from_dict(peer_data['hand'])

            if peer_data.get('is_host', False):
                # Trust host for scoreboard and authority decisions
                if 'score' in peer_data:
                    score = peer_data['score']
                if 'authority_owner' in peer_data:
                    authority_owner = peer_data['authority_owner'] or 'host'
                # On explicit reset from host, snap to host state unconditionally
                if peer_data.get('authority_reset', False) and 'ball' in peer_data:
                    ball.from_dict(peer_data['ball'])
                    prev_ball_x, prev_ball_y = ball.x, ball.y
                    last_interaction_time = time.time()
                else:
                    # Only apply host ball state when host owns authority
                    if authority_owner != 'client' and 'ball' in peer_data:
                        ball.from_dict(peer_data['ball'])
                        prev_ball_x, prev_ball_y = ball.x, ball.y
                        last_interaction_time = time.time()
            else:
                # Message from client (host cares)
                if is_host:
                    # Track last time client sent anything (for timeout)
                    last_client_ball_rx_time = time.time()
                    if authority_owner == 'client' and 'ball' in peer_data:
                        candidate = peer_data['ball']
                        if ball_state_is_valid(candidate, prev_ball_x, prev_ball_y):
                            ball.from_dict(candidate)
                            prev_ball_x, prev_ball_y = ball.x, ball.y
                            last_interaction_time = time.time()
                        else:
                            # Revoke immediately on invalid state
                            authority_owner = 'host'
                            authority_reset_to_send = True
                            # Snap ball to a safe host-side center
                            ball.x = (cam_width // 2) if is_left_player else (cam_width + cam_width // 2)
                            ball.y = display_height // 2
                            ball.vx = 0; ball.vy = 0
                            prev_ball_x, prev_ball_y = ball.x, ball.y
                            last_authority_change_time = time.time()
                            print("Authority revoked due to invalid client state")

        # Host decides ownership transitions and anomalies
        if is_host:
            now = time.time()

            # Client timeout while client should own -> revoke but keep ball state (no snap)
            if authority_owner == 'client' and (now - last_client_ball_rx_time) > client_timeout:
                authority_owner = 'host'
                # Keep ball position/velocity; just change authority
                last_authority_change_time = now
                print("Authority revoked due to client timeout (kept ball state)")

            # If client owns but ball is not on client side (with margin) and not center band -> switch without snap
            if authority_owner == 'client' and not is_on_client_side_with_margin(ball.x) and not is_in_center_band(ball.x):
                authority_owner = 'host'
                # Keep ball position/velocity; just change authority
                last_authority_change_time = now
                print("Authority returned to host due to side mismatch (kept ball state)")

            # Natural ownership switch by half with hysteresis/dwell/cooldown (no reset)
            new_desired = None
            if is_on_client_side_with_margin(ball.x):
                new_desired = 'client'
            elif is_on_host_side_with_margin(ball.x):
                new_desired = 'host'
            # Update desired side tracking
            if new_desired != desired_side:
                desired_side = new_desired
                desired_since = now
            # Perform switch if stable and cooled down
            if desired_side is not None and desired_side != authority_owner:
                if (now - last_authority_change_time) >= authority_cooldown and (now - desired_since) >= switch_dwell_time:
                    authority_owner = desired_side
                    last_authority_change_time = now
                    if authority_owner == 'client':
                        # Give client time to start sending before timeout triggers
                        last_client_ball_rx_time = now
                    print(f"Authority changed to {authority_owner} (hysteresis)")

            # Host-controlled idle timeout respawn (snap + reset)
            time_since_interaction = now - last_interaction_time
            ball_velocity = abs(ball.vx) + abs(ball.vy)
            if time_since_interaction > authority_timeout and ball_velocity < 0.5:
                ball.x = (cam_width // 2) if is_left_player else (cam_width + cam_width // 2)
                ball.y = display_height // 2
                ball.vx = 0; ball.vy = 0
                prev_ball_x, prev_ball_y = ball.x, ball.y
                authority_owner = 'host'
                authority_reset_to_send = True
                last_interaction_time = now
                last_authority_change_time = now
                print("HOST TIMEOUT RESET - Ball respawned")

        # Local control: simulate whenever you have authority (no side gating)
        local_has_control = (authority_owner == ('host' if is_host else 'client'))
        simulate_locally = local_has_control

        # Physics simulation only on the side with local control
        if simulate_locally:
            # Check collision with my hand
            if my_hand.check_collision(ball):
                dx = ball.x - my_hand.x
                dy = ball.y - my_hand.y
                dist = np.hypot(dx, dy)
                if dist > 0:
                    dx /= dist; dy /= dist
                    impulse_strength = 2.0
                    ball.apply_force(hand_vx * impulse_strength + dx * 5,
                                     hand_vy * impulse_strength + dy * 5)
                    overlap = (my_hand.radius + ball.radius) - dist
                    ball.x += dx * (overlap + 3)
                    ball.y += dy * (overlap + 3)
                last_interaction_time = time.time()

            # Check collision with peer hand
            if peer_hand.check_collision(ball):
                dx = ball.x - peer_hand.x
                dy = ball.y - peer_hand.y
                dist = np.hypot(dx, dy)
                if dist > 0:
                    dx /= dist; dy /= dist
                    overlap = (peer_hand.radius + ball.radius) - dist
                    ball.x += dx * (overlap + 3)
                    ball.y += dy * (overlap + 3)
                    ball.apply_force(dx * 5, dy * 5)
                last_interaction_time = time.time()

            # Update ball physics
            ball.update(display_width, display_height, floor_height)
            prev_ball_x, prev_ball_y = ball.x, ball.y

        # Host always scores (even when client has authority)
        if is_host:
            if ball.x - ball.radius <= 0:
                score['right'] += 1
                # Respawn on left (loser's side)
                ball.x = cam_width // 2
                ball.y = display_height // 2
                ball.vx = 0; ball.vy = 0
                prev_ball_x, prev_ball_y = ball.x, ball.y
                # Authority by respawn side
                authority_owner = 'host' if is_left_player else 'client'
                # Prevent immediate timeout after giving client authority
                if authority_owner == 'client':
                    last_client_ball_rx_time = time.time()
                authority_reset_to_send = True
                last_interaction_time = time.time()
                last_authority_change_time = time.time()
                print(f"RIGHT SCORES! {score['left']} - {score['right']}")
            elif ball.x + ball.radius >= display_width:
                score['left'] += 1
                # Respawn on right (loser's side)
                ball.x = cam_width + cam_width // 2
                ball.y = display_height // 2
                ball.vx = 0; ball.vy = 0
                prev_ball_x, prev_ball_y = ball.x, ball.y
                authority_owner = 'host' if not is_left_player else 'client'
                # Prevent immediate timeout after giving client authority
                if authority_owner == 'client':
                    last_client_ball_rx_time = time.time()
                authority_reset_to_send = True
                last_interaction_time = time.time()
                last_authority_change_time = time.time()
                print(f"LEFT SCORES! {score['left']} - {score['right']}")

        # Send game state (host authoritative about authority_owner, include resets)
        if network.connected:
            if is_host:
                network.send_game_state(
                    ball, my_hand, score,
                    has_authority=(authority_owner == 'host'),
                    is_host=True,
                    authority_reset=authority_reset_to_send,
                    authority_owner=authority_owner
                )
            else:
                # Client sends ball to host only when it believes it owns control
                network.send_game_state(
                    ball, my_hand, score,
                    has_authority=(authority_owner == 'client'),
                    is_host=False,
                    authority_reset=False,
                    authority_owner=authority_owner
                )

        # Draw ball/hands/UI
        ball.draw(canvas)
        my_hand.draw(canvas)
        peer_hand.draw(canvas)

        mode_text = "HOST" if is_host else "CLIENT"
        authority_text = " [CONTROLLING BALL]" if simulate_locally else " [WATCHING]"
        status_text = f"{mode_text}{authority_text} - {'CONNECTED' if network.connected else 'WAITING...'}"
        status_color = (0, 255, 0) if network.connected else (0, 165, 255)
        cv2.putText(canvas, status_text, (10, display_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        score_text = f"{score['left']} - {score['right']}"
        cv2.putText(canvas, score_text, (display_width // 2 - 40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if not network.connected:
            cv2.putText(canvas, "Waiting for opponent...", (display_width // 2 - 150, display_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('Multiplayer Hand Ball Game', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    network.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
