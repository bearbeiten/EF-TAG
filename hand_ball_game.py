import cv2
import mediapipe as mp
import numpy as np
from collections import deque

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
    
    # Initialize ball
    ball = Ball(width // 2, height // 2)
    
    # Hand position history for velocity calculation
    hand_positions = deque(maxlen=5)
    hand_polygon_history = deque(maxlen=3)
    
    print("Controls:")
    print("- Move your hand to interact with the ball")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset ball position")
    
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
                    
                    # Add to position history
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
                # Calculate push direction (from hand center to ball)
                if hand_center_x is not None:
                    dx = ball.x - hand_center_x
                    dy = ball.y - hand_center_y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist > 0:
                        # Normalize direction
                        dx /= dist
                        dy /= dist
                        
                        # Apply impulse based on hand velocity and penetration
                        impulse_strength = 2.0
                        ball.apply_force(hand_vx * impulse_strength + dx * 5, 
                                       hand_vy * impulse_strength + dy * 5)
                        
                        # Push ball out of hand
                        ball.x += dx * 3
                        ball.y += dy * 3
                
                # Visual feedback
                cv2.circle(frame, (int(ball.x), int(ball.y)), ball.radius + 5, (0, 0, 255), 3)
        
        # Update and draw ball
        ball.update(width, height)
        ball.draw(frame)
        
        # Draw info
        cv2.putText(frame, f"Ball Vel: ({ball.vx:.1f}, {ball.vy:.1f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if hand_center_x is not None:
            cv2.putText(frame, f"Hand Vel: ({hand_vx:.1f}, {hand_vy:.1f})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", 
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Hand Ball Game', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            ball.x = width // 2
            ball.y = height // 2
            ball.vx = 0
            ball.vy = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
