import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import random
import time

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
    
    def update(self, width, height):
        # Apply gravity
        self.vy += self.gravity
        
        # Apply damping
        self.vx *= self.damping
        self.vy *= self.damping
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Wall collisions
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * self.bounce_damping
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx = -self.vx * self.bounce_damping
        
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

class Target:
    def __init__(self, x, y, radius=40, points=10, color=(255, 0, 0)):
        self.x = x
        self.y = y
        self.radius = radius
        self.points = points
        self.color = color
        self.hit = False
        self.hit_time = 0
    
    def check_collision(self, ball):
        dx = self.x - ball.x
        dy = self.y - ball.y
        distance = np.sqrt(dx**2 + dy**2)
        return distance < (self.radius + ball.radius)
    
    def draw(self, frame, current_time):
        if self.hit:
            # Fade out animation
            alpha = max(0, 1 - (current_time - self.hit_time) * 2)
            if alpha > 0:
                overlay = frame.copy()
                cv2.circle(overlay, (int(self.x), int(self.y)), 
                          int(self.radius * (1 + (1-alpha))), self.color, -1)
                cv2.addWeighted(overlay, alpha * 0.5, frame, 1, 0, frame)
            return alpha > 0
        else:
            # Draw target
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius//2, (255, 255, 255), -1)
            # Draw points
            cv2.putText(frame, f"+{self.points}", (int(self.x)-15, int(self.y)+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            return True

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.score = 0
        self.high_score = 0
        self.targets = []
        self.game_time = 60  # 60 seconds
        self.start_time = time.time()
        self.level = 1
        self.targets_hit = 0
        self.spawn_timer = 0
        self.spawn_interval = 2.0  # seconds between spawns
    
    def spawn_target(self):
        # Avoid spawning too close to edges
        margin = 80
        x = random.randint(margin, self.width - margin)
        y = random.randint(margin, self.height - margin)
        
        # Different target types based on level
        rand = random.random()
        if rand < 0.6:  # 60% - Normal target
            target = Target(x, y, radius=40, points=10, color=(255, 100, 0))
        elif rand < 0.85:  # 25% - Small target (more points)
            target = Target(x, y, radius=25, points=25, color=(0, 100, 255))
        else:  # 15% - Large target (less points)
            target = Target(x, y, radius=60, points=5, color=(100, 255, 100))
        
        self.targets.append(target)
    
    def update(self, ball):
        current_time = time.time()
        elapsed = current_time - self.start_time
        remaining_time = max(0, self.game_time - elapsed)
        
        # Spawn targets
        if remaining_time > 0:
            self.spawn_timer += 1/30  # Assuming ~30 FPS
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_target()
                self.spawn_timer = 0
                # Increase difficulty
                if self.level < 5:
                    self.spawn_interval = max(1.0, 2.0 - self.level * 0.2)
        
        # Check collisions
        for target in self.targets[:]:
            if not target.hit and target.check_collision(ball):
                target.hit = True
                target.hit_time = current_time
                self.score += target.points
                self.targets_hit += 1
                
                # Level up every 10 targets
                new_level = (self.targets_hit // 10) + 1
                if new_level > self.level:
                    self.level = new_level
        
        # Remove faded targets
        self.targets = [t for t in self.targets if not t.hit or 
                       (current_time - t.hit_time) < 0.5]
        
        # Update high score
        if self.score > self.high_score:
            self.high_score = self.score
        
        return remaining_time
    
    def draw(self, frame, remaining_time):
        current_time = time.time()
        
        # Draw targets
        for target in self.targets:
            target.draw(frame, current_time)
        
        # Draw UI
        ui_y = 30
        cv2.putText(frame, f"Score: {self.score}", (10, ui_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"High Score: {self.high_score}", (10, ui_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Level: {self.level}", (10, ui_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Timer
        timer_text = f"Time: {int(remaining_time)}s"
        timer_color = (0, 255, 0) if remaining_time > 10 else (0, 0, 255)
        cv2.putText(frame, timer_text, (self.width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)
        
        # Game over screen
        if remaining_time <= 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.width//4, self.height//3),
                         (3*self.width//4, 2*self.height//3), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "GAME OVER!", (self.width//2 - 120, self.height//2 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", (self.width//2 - 100, self.height//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to restart", (self.width//2 - 130, self.height//2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    def reset(self):
        self.score = 0
        self.targets = []
        self.start_time = time.time()
        self.level = 1
        self.targets_hit = 0
        self.spawn_timer = 0
        self.spawn_interval = 2.0

def get_hand_polygon(hand_landmarks, width, height):
    """Extract hand contour points from all landmarks"""
    # Get all landmark points
    points = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # Create convex hull around all points
    hull = cv2.convexHull(points)
    return hull

def check_collision_with_polygon(ball, hand_polygon):
    """Check if ball collides with hand polygon"""
    if hand_polygon is None or len(hand_polygon) == 0:
        return False, None, None
    
    # Check if ball center is inside polygon
    ball_center = (int(ball.x), int(ball.y))
    dist = cv2.pointPolygonTest(hand_polygon, ball_center, True)
    
    # If distance is positive (inside) or close to edge
    if dist > -ball.radius:
        # Find closest point on polygon edge
        min_dist = float('inf')
        closest_point = None
        
        for i in range(len(hand_polygon)):
            p1 = hand_polygon[i][0]
            p2 = hand_polygon[(i + 1) % len(hand_polygon)][0]
            
            # Calculate distance from ball to line segment
            line_dist = point_to_segment_distance(ball_center, p1, p2)
            if line_dist < min_dist:
                min_dist = line_dist
                closest_point = closest_point_on_segment(ball_center, p1, p2)
        
        return True, closest_point, dist
    
    return False, None, None

def point_to_segment_distance(point, seg_a, seg_b):
    """Calculate distance from point to line segment"""
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
    """Find closest point on line segment to given point"""
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return seg_a
    
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2)))
    return (int(ax + t * dx), int(ay + t * dy))

def main():
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
    
    # Initialize ball and game
    ball = Ball(width // 2, height // 2)
    game = Game(width, height)
    
    # Hand position history for velocity calculation
    hand_positions = deque(maxlen=5)
    hand_polygon_history = deque(maxlen=3)
    
    print("=== HAND BALL TARGET GAME ===")
    print("Hit targets with the ball to score points!")
    print("Different colors = different points")
    print("- Orange (large): 10 points")
    print("- Blue (small): 25 points") 
    print("- Green (huge): 5 points")
    print("\nControls:")
    print("- Move your hand to push the ball")
    print("- Press SPACE to restart after game over")
    print("- Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(rgb_frame)
        
        hand_polygon = None
        hand_center_x, hand_center_y = None, None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get hand polygon (convex hull of all landmarks)
                hand_polygon = get_hand_polygon(hand_landmarks, width, height)
                
                # Draw hand segmentation overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [hand_polygon], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [hand_polygon], True, (0, 255, 0), 2)
                
                # Calculate hand center (centroid of polygon)
                M = cv2.moments(hand_polygon)
                if M["m00"] != 0:
                    hand_center_x = int(M["m10"] / M["m00"])
                    hand_center_y = int(M["m01"] / M["m00"])
                    
                    cv2.circle(frame, (hand_center_x, hand_center_y), 10, (255, 0, 0), -1)
                    hand_positions.append((hand_center_x, hand_center_y))
                
                hand_polygon_history.append(hand_polygon)
        
        # Calculate hand velocity
        hand_vx, hand_vy = 0, 0
        if len(hand_positions) >= 2:
            hand_vx = (hand_positions[-1][0] - hand_positions[0][0]) / len(hand_positions)
            hand_vy = (hand_positions[-1][1] - hand_positions[0][1]) / len(hand_positions)
        
        # Check collision with hand polygon
        if hand_polygon is not None:
            collision, closest_point, penetration_dist = check_collision_with_polygon(ball, hand_polygon)
            
            if collision:
                if hand_center_x is not None:
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
        
        # Update ball and game
        ball.update(width, height)
        remaining_time = game.update(ball)
        
        # Draw everything
        game.draw(frame, remaining_time)
        ball.draw(frame)
        
        # Display frame
        cv2.imshow('Hand Ball Target Game', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to restart
            ball.x = width // 2
            ball.y = height // 2
            ball.vx = 0
            ball.vy = 0
            game.reset()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
