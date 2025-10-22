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
    
    def update(self, width, height, floor_height=100):
        # Apply gravity
        self.vy += self.gravity
        
        # Apply damping
        self.vx *= self.damping
        self.vy *= self.damping
        
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
    
    def send_game_state(self, ball, hand, score, has_authority, is_host, authority_reset=False):
        """Send complete game state to peer"""
        if self.connected:
            self._send_message({
                'type': 'game_state',
                'ball': ball.to_dict(),
                'hand': hand.to_dict(),
                'score': score,
                'has_authority': has_authority,
                'is_host': is_host,
                'authority_reset': authority_reset
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
    
    # Initialize hands
    my_hand = Hand(radius=50)
    peer_hand = Hand(radius=50)
    
    score = {'left': 0, 'right': 0}
    
    hand_positions = deque(maxlen=5)
    
    # Start with left player having authority
    is_ball_authority = is_left_player
    last_ball_side = 'left'  # Track which side ball is on
    
    # Authority timeout tracking
    last_authority_transfer_time = time.time()
    authority_timeout = 15.0  # seconds
    last_ball_velocity = 0
    authority_reset_flag = False
    
    print("\n=== MULTIPLAYER HAND BALL GAME (SPLIT SCREEN) ===")
    print(f"You are: {'LEFT' if is_left_player else 'RIGHT'} player")
    print(f"Mode: {'SERVER' if is_host else 'CLIENT'}")
    print("Hit the ball to the opponent's side!")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
        peer_has_authority = False
        peer_is_host = False
        
        if peer_data:
            # Update peer hand
            if 'hand' in peer_data:
                peer_hand.from_dict(peer_data['hand'])
            
            # Check if peer has authority and is host
            peer_has_authority = peer_data.get('has_authority', False)
            peer_is_host = peer_data.get('is_host', False)
            
            # Check for authority reset from host
            if peer_data.get('authority_reset', False) and peer_is_host:
                # Host has reset authority, accept their ball state
                if 'ball' in peer_data:
                    ball.from_dict(peer_data['ball'])
                is_ball_authority = peer_has_authority and not is_host
                last_authority_transfer_time = time.time()
                
                # Determine ball side from position
                if ball.x < cam_width:
                    last_ball_side = 'left'
                else:
                    last_ball_side = 'right'
                
                print("Authority reset by host - ball position synchronized")
            elif peer_has_authority and 'ball' in peer_data:
                # Normal update from peer with authority
                ball.from_dict(peer_data['ball'])
                last_authority_transfer_time = time.time()  # Reset timeout
            
            # Host has authority over score - client always accepts host's score
            if 'score' in peer_data:
                if peer_is_host:
                    # Peer is host, accept their score as truth
                    score = peer_data['score']
                elif not is_host:
                    # Neither is host (shouldn't happen), but just in case
                    score = peer_data['score']
        
        # Determine which side the ball is on
        if ball.x < cam_width:
            current_ball_side = 'left'
        else:
            current_ball_side = 'right'
        
        # Calculate ball velocity
        current_ball_velocity = abs(ball.vx) + abs(ball.vy)
        
        # Reset authority reset flag
        authority_reset_flag = False
        
        # Check for authority timeout (ball stuck, no movement)
        time_since_transfer = time.time() - last_authority_transfer_time
        if time_since_transfer > authority_timeout and current_ball_velocity < 0.5:
            # Ball hasn't moved in a while and no authority transfer
            if is_host:
                # Host claims authority and resets ball to host's side
                is_ball_authority = True
                authority_reset_flag = True
                
                # Spawn ball on host's side
                if is_left_player:
                    ball.x = cam_width // 2
                    last_ball_side = 'left'
                else:
                    ball.x = cam_width + cam_width // 2
                    last_ball_side = 'right'
                
                ball.y = display_height // 2
                ball.vx = 0
                ball.vy = 0
                last_authority_transfer_time = time.time()
                print("Authority timeout - host claiming ball control and resetting position")
        
        # Transfer authority if ball changed sides (only if no reset)
        if current_ball_side != last_ball_side and not authority_reset_flag:
            last_ball_side = current_ball_side
            last_authority_transfer_time = time.time()  # Reset timeout
            
            # Authority transfers to player on ball's side
            if (current_ball_side == 'left' and is_left_player) or \
               (current_ball_side == 'right' and not is_left_player):
                is_ball_authority = True
                print(f"Authority transferred to you (ball on {current_ball_side} side)")
            else:
                is_ball_authority = False
                print(f"Authority transferred to peer (ball on {current_ball_side} side)")
        
        # Only update ball physics if this player has authority
        if is_ball_authority:
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
                
                last_authority_transfer_time = time.time()  # Reset timeout on interaction
            
            # Check collision with peer hand
            if peer_hand.check_collision(ball):
                dx = ball.x - peer_hand.x
                dy = ball.y - peer_hand.y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    
                    # Push ball away from peer hand
                    overlap = (peer_hand.radius + ball.radius) - dist
                    ball.x += dx * (overlap + 3)
                    ball.y += dy * (overlap + 3)
                    
                    # Apply some force
                    ball.apply_force(dx * 5, dy * 5)
                
                last_authority_transfer_time = time.time()  # Reset timeout on interaction
            
            # Update ball physics
            ball.update(display_width, display_height, floor_height)
            
            # Check scoring - ANY player with authority can detect scoring, but only host updates score
            scored = False
            scoring_side = None
            
            if ball.x - ball.radius <= 0:
                # Ball touched left edge - right player scores
                scoring_side = 'right'
                scored = True
            elif ball.x + ball.radius >= display_width:
                # Ball touched right edge - left player scores
                scoring_side = 'left'
                scored = True
            
            # If scoring detected
            if scored:
                # Only host updates the actual score
                if is_host:
                    if scoring_side == 'left':
                        score['left'] += 1
                        print(f"Left player scores! Score: {score['left']} - {score['right']}")
                        # Spawn on right side (loser's side)
                        ball.x = cam_width + cam_width // 2
                        last_ball_side = 'right'
                        is_ball_authority = not is_left_player
                    else:  # scoring_side == 'right'
                        score['right'] += 1
                        print(f"Right player scores! Score: {score['left']} - {score['right']}")
                        # Spawn on left side (loser's side)
                        ball.x = cam_width // 2
                        last_ball_side = 'left'
                        is_ball_authority = is_left_player
                    
                    ball.y = display_height // 2
                    ball.vx = 0
                    ball.vy = 0
                    last_authority_transfer_time = time.time()
                    authority_reset_flag = True  # Signal reset to client
                else:
                    # Client detected scoring but can't update score
                    # Just report it and let host handle it
                    print(f"Detected scoring by {scoring_side} player - waiting for host confirmation")
                    # Reset ball locally and let host sync
                    if scoring_side == 'left':
                        ball.x = cam_width + cam_width // 2
                        last_ball_side = 'right'
                    else:
                        ball.x = cam_width // 2
                        last_ball_side = 'left'
                    
                    ball.y = display_height // 2
                    ball.vx = 0
                    ball.vy = 0
                    is_ball_authority = False  # Give up authority, wait for host
                    last_authority_transfer_time = time.time()
        
        last_ball_velocity = current_ball_velocity
        
        # Send game state to peer
        if network.connected:
            network.send_game_state(ball, my_hand, score, is_ball_authority, is_host, authority_reset_flag)
        
        # Draw ball
        ball.draw(canvas)
        
        # Draw my hand on my side
        my_hand.draw(canvas)
        
        # Draw peer hand on their side
        peer_hand.draw(canvas)
        
        # Draw UI
        mode_text = "SERVER" if is_host else "CLIENT"
        authority_indicator = " [BALL CONTROL]" if is_ball_authority else ""
        status_text = f"{mode_text}{authority_indicator} - {'CONNECTED' if network.connected else 'WAITING...'}"
        status_color = (0, 255, 0) if network.connected else (0, 165, 255)
        cv2.putText(canvas, status_text, (10, display_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw score
        score_text = f"{score['left']} - {score['right']}"
        cv2.putText(canvas, score_text, (display_width // 2 - 40, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Check connection status
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
