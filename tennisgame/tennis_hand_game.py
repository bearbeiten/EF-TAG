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
        self.center_line = None  # Will be set to cam_width for reference
        
        # For client-side interpolation
        self.target_x = x
        self.target_y = y
        self.target_vx = 0
        self.target_vy = 0
        self.interpolation_speed = 0.6  # Higher = faster catch-up (increased from 0.3)
    
    def update(self, width, height, floor_height=100):
        # Apply gravity
        self.vy += self.gravity
        
        # Apply damping
        self.vx *= self.damping
        self.vy *= self.damping
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Wall collisions - only bounce at outer edges (0 and width)
        # Allow free passage through center line
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * self.bounce_damping  # Bounce right
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx = -abs(self.vx) * self.bounce_damping  # Bounce left
        
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
    
    def from_dict(self, data, interpolate=False):
        """Update ball state from network data"""
        if interpolate:
            # Set targets for interpolation
            self.target_x = data['x']
            self.target_y = data['y']
            self.target_vx = data['vx']
            self.target_vy = data['vy']
        else:
            # Direct update (for host)
            self.x = data['x']
            self.y = data['y']
            self.vx = data['vx']
            self.vy = data['vy']
            self.target_x = self.x
            self.target_y = self.y
            self.target_vx = self.vx
            self.target_vy = self.vy
    
    def interpolate_to_target(self):
        """Smoothly interpolate position towards target (for clients)"""
        # Calculate distance to target
        dist_x = abs(self.target_x - self.x)
        dist_y = abs(self.target_y - self.y)
        
        # If we're very close to target, snap to it
        if dist_x < 2 and dist_y < 2:
            self.x = self.target_x
            self.y = self.target_y
            self.vx = self.target_vx
            self.vy = self.target_vy
        else:
            # Interpolate position and velocity
            self.x += (self.target_x - self.x) * self.interpolation_speed
            self.y += (self.target_y - self.y) * self.interpolation_speed
            self.vx += (self.target_vx - self.vx) * self.interpolation_speed
            self.vy += (self.target_vy - self.vy) * self.interpolation_speed
    
    def predict_client_side(self, width, height, floor_height=100):
        """Client-side prediction: simulate physics with current velocity"""
        # Apply physics similar to update but gentler
        self.vy += self.gravity * 0.5  # Reduced gravity for prediction
        
        self.vx *= self.damping
        self.vy *= self.damping
        
        # Update position based on predicted velocity
        self.x += self.vx
        self.y += self.vy
        
        # Basic boundary checks (simplified, no bouncing)
        if self.x < self.radius:
            self.x = self.radius
            self.vx = 0
        elif self.x > width - self.radius:
            self.x = width - self.radius
            self.vx = 0
        
        if self.y < self.radius:
            self.y = self.radius
            self.vy = 0
        
        floor_y = height - floor_height
        if self.y > floor_y - self.radius:
            self.y = floor_y - self.radius
            self.vy = 0

class Hand:
    def __init__(self, radius=50):
        self.x = None
        self.y = None
        self.radius = radius
        self.is_active = False
        self.last_seen_time = 0
        self.timeout = 0.5  # seconds before hand disappears
    
    def update(self, center_x, center_y):
        """Update hand position when detected"""
        self.x = center_x
        self.y = center_y
        self.is_active = True
        self.last_seen_time = time.time()
    
    def check_timeout(self):
        """Check if hand should disappear due to timeout"""
        if self.is_active and (time.time() - self.last_seen_time) > self.timeout:
            self.is_active = False
    
    def draw(self, frame):
        """Draw the hand circle"""
        if self.is_active and self.x is not None and self.y is not None:
            # Draw semi-transparent filled circle
            overlay = frame.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), self.radius, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Draw outline
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (0, 255, 0), 3)
            
            # Draw center point
            cv2.circle(frame, (int(self.x), int(self.y)), 5, (255, 0, 0), -1)
    
    def check_collision(self, ball):
        """Check if hand collides with ball"""
        if not self.is_active or self.x is None or self.y is None:
            return False
        
        dx = self.x - ball.x
        dy = self.y - ball.y
        distance = np.sqrt(dx**2 + dy**2)
        return distance < (self.radius + ball.radius)
    
    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'is_active': self.is_active
        }
    
    def from_dict(self, data):
        if data:
            self.x = data.get('x')
            self.y = data.get('y')
            self.is_active = data.get('is_active', False)
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
        
        # Network throttling
        self.last_send_time = 0
        self.send_interval = 1.0 / 60.0  # Send at 60 FPS (increased from 30)
        self.last_hand_send_time = 0
        self.hand_send_interval = 1.0 / 60.0  # Send hand updates at 60 FPS
        
        # Thread safety
        self.data_lock = threading.Lock()
    
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
                        
                        # Optimize socket for low latency
                        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        
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
                
                # Optimize socket for low latency
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
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
            
            # Sanity check message length
            if message_length > 1000000:  # 1MB limit
                print(f"Message too large: {message_length}")
                return None
            
            # Receive the actual message
            message_data = b''
            while len(message_data) < message_length:
                chunk = sock.recv(min(8192, message_length - len(message_data)))
                if not chunk:
                    return None
                message_data += chunk
            
            return json.loads(message_data.decode('utf-8'))
        except socket.timeout:
            return None
        except Exception as e:
            # Only log unexpected errors
            if self.running and self.connected:
                print(f"Receive error: {e}")
            return None
    
    def _send_message(self, message, sock=None):
        """Send length-prefixed JSON message"""
        if sock is None:
            sock = self.connection
        
        if sock is None:
            return
        
        try:
            # Use separators to minimize JSON size
            data = json.dumps(message, separators=(',', ':')).encode('utf-8')
            length = len(data).to_bytes(4, 'big')
            sock.sendall(length + data)
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
    
    def _receive_loop(self):
        """Main receive loop for handling messages"""
        # Set socket to non-blocking mode with very short timeout
        if self.connection:
            self.connection.settimeout(0.05)  # 50ms timeout (reduced from 100ms)
        
        while self.running and self.connected:
            try:
                message = self._recv_message(self.connection)
                if message is None:
                    # Timeout or connection closed
                    continue
                
                if message['type'] == 'game_state':
                    with self.data_lock:
                        self.received_data = message
                        self.last_receive_time = time.time()
                elif message['type'] == 'ping':
                    self._send_message({'type': 'pong'})
                    
            except socket.timeout:
                # Normal timeout, continue
                continue
            except Exception as e:
                if self.running:
                    print(f"Receive loop error: {e}")
                self.connected = False
                break
    
    def send_game_state(self, ball, hand, score, has_authority, is_host, authority_reset=False, display_width=None):
        """Send complete game state to peer with throttling"""
        if not self.connected:
            return
        
        current_time = time.time()
        
        # Throttle game state sends (host sends ball state)
        if is_host and (current_time - self.last_send_time) < self.send_interval:
            return
        
        # For clients, only send hand updates at higher frequency
        if not is_host and (current_time - self.last_hand_send_time) < self.hand_send_interval:
            return
        
        message = {
            'type': 'game_state',
            'ball': ball.to_dict() if is_host else None,  # Only host sends ball
            'hand': hand.to_dict(),
            'score': score if is_host else None,  # Only host sends score
            'has_authority': has_authority,
            'is_host': is_host,
            'authority_reset': authority_reset,
            'timestamp': current_time
        }
        if display_width is not None:
            message['display_width'] = display_width
        
        self._send_message(message)
        
        if is_host:
            self.last_send_time = current_time
        else:
            self.last_hand_send_time = current_time
    
    def get_received_data(self):
        """Get received game state (thread-safe)"""
        with self.data_lock:
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
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display dimensions (double width for split screen)
    display_width = cam_width * 2
    display_height = cam_height
    
    # Floor settings
    floor_height = 100
    
    # Initialize ball (start on left side)
    ball = Ball(cam_width // 2, display_height // 2)
    ball.center_line = cam_width  # Set center line reference
    
    # Initialize hands
    my_hand = Hand(radius=50)
    peer_hand = Hand(radius=50)
    
    score = {'left': 0, 'right': 0}
    
    hand_positions = deque(maxlen=5)
    
    # Authority system:
    # Host is the only authority that simulates physics and decides authority.
    is_ball_authority = is_host  # local authority equals host status
    
    # Track previous ball X position for host-only logs if needed
    prev_ball_x = ball.x
    
    # Authority timeout tracking (host only)
    last_interaction_time = time.time()
    authority_timeout = 15.0  # seconds
    
    print("\n=== MULTIPLAYER HAND BALL GAME (SPLIT SCREEN) ===")
    print(f"You are: {'LEFT' if is_left_player else 'RIGHT'} player")
    print(f"Mode: {'SERVER (HOST)' if is_host else 'CLIENT'}")
    print("Hit the ball to the opponent's side!")
    print("Press 'q' to quit")
    
    frame_count = 0
    last_frame_time = time.time()
    target_fps = 60
    frame_delay = 1.0 / target_fps
    fps_display = 60.0
    fps_update_time = time.time()
    
    while cap.isOpened():
        frame_start_time = time.time()
        
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
        
        # Process hand detection
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_polygon = get_hand_polygon(hand_landmarks, cam_width, cam_height)
                
                # Calculate hand center
                M = cv2.moments(hand_polygon)
                if M["m00"] != 0:
                    hand_center_x = int(M["m10"] / M["m00"])
                    hand_center_y = int(M["m01"] / M["m00"])
                    
                    # Offset to full canvas position
                    if is_left_player:
                        canvas_x = hand_center_x
                    else:
                        canvas_x = cam_width + hand_center_x
                    
                    my_hand.update(canvas_x, hand_center_y)
                    hand_positions.append((canvas_x, hand_center_y))
        
        # Check hand timeout
        my_hand.check_timeout()
        
        # Calculate hand velocity
        hand_vx, hand_vy = 0, 0
        if len(hand_positions) >= 2:
            hand_vx = (hand_positions[-1][0] - hand_positions[0][0]) / len(hand_positions)
            hand_vy = (hand_positions[-1][1] - hand_positions[0][1]) / len(hand_positions)
        
        # Get peer data
        peer_data = network.get_received_data()
        
        if peer_data:
            # Always update peer hand
            if 'hand' in peer_data and peer_data['hand'] is not None:
                peer_hand.from_dict(peer_data['hand'])
            
            # If we receive data from the host, trust their ball state and score
            if peer_data.get('is_host', False):
                if 'score' in peer_data and peer_data['score'] is not None:
                    score = peer_data['score']
                if 'ball' in peer_data and peer_data['ball'] is not None:
                    # Host sends ball in full double-width coordinate system
                    # Use interpolation for smooth rendering on client
                    ball.from_dict(peer_data['ball'], interpolate=(not is_host))
                    if not is_host:
                        # For clients, just set the target
                        pass
                    else:
                        prev_ball_x = ball.x
                    last_interaction_time = time.time()
        
        # Host decides authority and simulates physics; clients only render
        # Remove local authority switching on center crossing.
        # Previous crossing-based authority logic removed in favor of host authority.
        
        # Check for authority timeout (host only)
        time_since_interaction = time.time() - last_interaction_time
        ball_velocity = abs(ball.vx) + abs(ball.vy)
        
        if is_host and time_since_interaction > authority_timeout and ball_velocity < 0.5:
            # Host reclaims authority and resets ball
            is_ball_authority = True  # host remains authority
            if is_left_player:
                ball.x = cam_width // 2
            else:
                ball.x = cam_width + cam_width // 2
            ball.y = display_height // 2
            ball.vx = 0
            ball.vy = 0
            prev_ball_x = ball.x
            last_interaction_time = time.time()
            print("HOST TIMEOUT RESET - Ball respawned")
        
        # Host simulates physics, client interpolates with prediction
        if is_host:
            # Check collision with my hand
            if my_hand.check_collision(ball):
                dx = ball.x - my_hand.x
                dy = ball.y - my_hand.y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    
                    impulse_strength = 2.0
                    ball.apply_force(hand_vx * impulse_strength + dx * 5, 
                                   hand_vy * impulse_strength + dy * 5)
                    
                    # Push ball out of hand
                    overlap = (my_hand.radius + ball.radius) - dist
                    ball.x += dx * (overlap + 3)
                    ball.y += dy * (overlap + 3)
                
                last_interaction_time = time.time()
            
            # Check collision with peer hand
            if peer_hand.check_collision(ball):
                dx = ball.x - peer_hand.x
                dy = ball.y - peer_hand.y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    
                    overlap = (peer_hand.radius + ball.radius) - dist
                    ball.x += dx * (overlap + 3)
                    ball.y += dy * (overlap + 3)
                    ball.apply_force(dx * 5, dy * 5)
                
                last_interaction_time = time.time()
            
            # Update ball physics (host only)
            ball.update(display_width, display_height, floor_height)
        else:
            # Client: Use hybrid approach
            # Interpolate towards server position while predicting movement
            ball.interpolate_to_target()
            # Add client-side prediction for smoother movement
            # Only predict if we have meaningful velocity
            if abs(ball.vx) > 0.1 or abs(ball.vy) > 0.1:
                ball.predict_client_side(display_width, display_height, floor_height)
            
            # Update score (host only)
            if ball.x - ball.radius <= 0:
                score['right'] += 1
                # Respawn on left (loser's side)
                ball.x = cam_width // 2
                ball.y = display_height // 2
                ball.vx = 0
                ball.vy = 0
                prev_ball_x = ball.x
                is_ball_authority = True  # host remains authority
                last_interaction_time = time.time()
                print(f"RIGHT SCORES! {score['left']} - {score['right']}")
            elif ball.x + ball.radius >= display_width:
                score['left'] += 1
                # Respawn on right (loser's side)
                ball.x = cam_width + cam_width // 2
                ball.y = display_height // 2
                ball.vx = 0
                ball.vy = 0
                prev_ball_x = ball.x
                is_ball_authority = True  # host remains authority
                last_interaction_time = time.time()
                print(f"LEFT SCORES! {score['left']} - {score['right']}")
        
        # Send game state to peer (host authoritative)
        if network.connected:
            # has_authority = True for host, False for client
            network.send_game_state(ball, my_hand, score, is_host, is_host, False, display_width)
        
        # Draw ball
        ball.draw(canvas)
        
        # Draw hands
        my_hand.draw(canvas)
        peer_hand.draw(canvas)
        
        # Draw UI
        mode_text = "HOST" if is_host else "CLIENT"
        authority_text = " [CONTROLLING BALL]" if is_host else " [WATCHING]"
        status_text = f"{mode_text}{authority_text} - {'CONNECTED' if network.connected else 'WAITING...'}"
        status_color = (0, 255, 0) if network.connected else (0, 165, 255)
        cv2.putText(canvas, status_text, (10, display_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Calculate and display FPS
        current_time = time.time()
        if current_time - fps_update_time > 0.5:  # Update FPS display every 0.5 seconds
            fps_display = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 60
            fps_update_time = current_time
        cv2.putText(canvas, f"FPS: {int(fps_display)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw score
        score_text = f"{score['left']} - {score['right']}"
        cv2.putText(canvas, score_text, (display_width // 2 - 40, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Check connection status
        if not network.connected:
            cv2.putText(canvas, "Waiting for opponent...", (display_width // 2 - 150, display_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('Multiplayer Hand Ball Game', canvas)
        
        # Frame rate control for consistent timing
        frame_elapsed = time.time() - frame_start_time
        wait_time = max(1, int((frame_delay - frame_elapsed) * 1000))
        
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        
        # Track actual FPS (optional, for debugging)
        last_frame_time = time.time()
    
    network.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
