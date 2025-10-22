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
    
    def update(self, width, height, is_left_player):
        # Apply gravity
        self.vy += self.gravity
        
        # Apply damping
        self.vx *= self.damping
        self.vy *= self.damping
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Wall collisions (left/right handled differently for multiplayer)
        if is_left_player:
            # Left player - only bounce on left side
            if self.x - self.radius < 0:
                self.x = self.radius
                self.vx = -self.vx * self.bounce_damping
        else:
            # Right player - only bounce on right side
            if self.x + self.radius > width:
                self.x = width - self.radius
                self.vx = -self.vx * self.bounce_damping
        
        # Top and bottom walls
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy * self.bounce_damping
        elif self.y + self.radius > height:
            self.y = height - self.radius
            self.vy = -self.vy * self.bounce_damping
    
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
        
        self.received_ball = None
        self.last_receive_time = 0
        
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
                
                if message['type'] == 'ball_transfer':
                    self.received_ball = message['ball']
                    self.last_receive_time = time.time()
                elif message['type'] == 'ping':
                    self._send_message({'type': 'pong'})
                    
            except Exception as e:
                if self.running:
                    print(f"Receive loop error: {e}")
                self.connected = False
                break
    
    def send_ball(self, ball):
        """Send ball to peer"""
        if self.connected:
            self._send_message({
                'type': 'ball_transfer',
                'ball': ball.to_dict()
            })
    
    def get_received_ball(self):
        """Get received ball data"""
        ball = self.received_ball
        self.received_ball = None
        return ball
    
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

def check_collision_with_polygon(ball, hand_polygon):
    if hand_polygon is None or len(hand_polygon) == 0:
        return False, None, None
    
    ball_center = (int(ball.x), int(ball.y))
    dist = cv2.pointPolygonTest(hand_polygon, ball_center, True)
    
    if dist > -ball.radius:
        min_dist = float('inf')
        closest_point = None
        
        for i in range(len(hand_polygon)):
            p1 = hand_polygon[i][0]
            p2 = hand_polygon[(i + 1) % len(hand_polygon)][0]
            
            line_dist = point_to_segment_distance(ball_center, p1, p2)
            if line_dist < min_dist:
                min_dist = line_dist
                closest_point = closest_point_on_segment(ball_center, p1, p2)
        
        return True, closest_point, dist
    
    return False, None, None

def point_to_segment_distance(point, seg_a, seg_b):
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return np.sqrt((px - ax)**2 + (py - ay)**2)
    
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def closest_point_on_segment(point, seg_a, seg_b):
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return seg_a
    
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2)))
    return (int(ax + t * dx), int(ay + t * dy))

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize ball
    if is_left_player:
        ball = Ball(width // 4, height // 2)
    else:
        ball = Ball(3 * width // 4, height // 2)
    
    has_ball = True
    score = {'left': 0, 'right': 0}
    
    hand_positions = deque(maxlen=5)
    
    print("\n=== MULTIPLAYER HAND BALL GAME (TCP) ===")
    print(f"You are: {'LEFT' if is_left_player else 'RIGHT'} player")
    print(f"Mode: {'SERVER' if is_host else 'CLIENT'}")
    print("Send the ball to the other side to score!")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw dividing line
        line_color = (0, 255, 0) if network.connected else (0, 0, 255)
        if is_left_player:
            cv2.line(frame, (width - 50, 0), (width - 50, height), line_color, 3)
            cv2.putText(frame, "SEND -->", (width - 140, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        else:
            cv2.line(frame, (50, 0), (50, height), line_color, 3)
            cv2.putText(frame, "<-- SEND", (60, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        
        # Check for received ball
        received_ball_data = network.get_received_ball()
        if received_ball_data and not has_ball:
            ball.from_dict(received_ball_data)
            # Flip ball position for opposite screen
            if is_left_player:
                ball.x = 100
                ball.vx = abs(ball.vx)
            else:
                ball.x = width - 100
                ball.vx = -abs(ball.vx)
            has_ball = True
            print("Ball received!")
        
        # Process hand detection
        results = hands.process(rgb_frame)
        
        hand_polygon = None
        hand_center_x, hand_center_y = None, None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_polygon = get_hand_polygon(hand_landmarks, width, height)
                
                overlay = frame.copy()
                cv2.fillPoly(overlay, [hand_polygon], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [hand_polygon], True, (0, 255, 0), 2)
                
                M = cv2.moments(hand_polygon)
                if M["m00"] != 0:
                    hand_center_x = int(M["m10"] / M["m00"])
                    hand_center_y = int(M["m01"] / M["m00"])
                    
                    cv2.circle(frame, (hand_center_x, hand_center_y), 10, (255, 0, 0), -1)
                    hand_positions.append((hand_center_x, hand_center_y))
        
        # Calculate hand velocity
        hand_vx, hand_vy = 0, 0
        if len(hand_positions) >= 2:
            hand_vx = (hand_positions[-1][0] - hand_positions[0][0]) / len(hand_positions)
            hand_vy = (hand_positions[-1][1] - hand_positions[0][1]) / len(hand_positions)
        
        # Handle ball physics and collision only if we have the ball
        if has_ball:
            if hand_polygon is not None:
                collision, _, _ = check_collision_with_polygon(ball, hand_polygon)
                
                if collision and hand_center_x is not None:
                    dx = ball.x - hand_center_x
                    dy = ball.y - hand_center_y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist > 0:
                        dx /= dist
                        dy /= dist
                        
                        impulse_strength = 2.0
                        ball.apply_force(hand_vx * impulse_strength + dx * 5, 
                                       hand_vy * impulse_strength + dy * 5)
                        ball.x += dx * 3
                        ball.y += dy * 3
                    
                    cv2.circle(frame, (int(ball.x), int(ball.y)), ball.radius + 5, (0, 0, 255), 3)
            
            ball.update(width, height, is_left_player)
            
            # Check if ball crossed to other side
            if is_left_player and ball.x > width - 50:
                if network.connected:
                    network.send_ball(ball)
                    has_ball = False
                    score['left'] += 1
                    print("Ball sent to opponent!")
                else:
                    ball.x = width - 50
                    ball.vx = -abs(ball.vx)
            elif not is_left_player and ball.x < 50:
                if network.connected:
                    network.send_ball(ball)
                    has_ball = False
                    score['right'] += 1
                    print("Ball sent to opponent!")
                else:
                    ball.x = 50
                    ball.vx = abs(ball.vx)
            
            ball.draw(frame)
        
        # Draw UI
        mode_text = "SERVER" if is_host else "CLIENT"
        status_text = f"{mode_text} - {'CONNECTED' if network.connected else 'WAITING...'}"
        status_color = (0, 255, 0) if network.connected else (0, 165, 255)
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(frame, f"Score: {score['left']} - {score['right']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        position_text = "LEFT Player" if is_left_player else "RIGHT Player"
        cv2.putText(frame, position_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        if not has_ball:
            cv2.putText(frame, "Waiting for ball...", (width // 2 - 100, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        # Check connection status
        if not network.connected and is_host:
            cv2.putText(frame, "Waiting for player to connect...", (width // 2 - 150, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Multiplayer Hand Ball Game (TCP)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    network.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
