import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from quiz_generator import generate_quiz_questions

# GestureRecognizer-Objekt erstellen basieren auf heruntergeladenem Modell
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Initialize MediaPipe drawing utilities for landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Game settings
ANSWER_TIME = 10  # seconds to answer each question
POINTS_PER_CORRECT = 1


class QuizGame:
    def __init__(self, questions, answer_time=10):
        self.questions = questions
        self.answer_time = answer_time
        self.current_question = 0
        self.player1_score = 0
        self.player2_score = 0
        self.question_start_time = None
        self.player1_answered = False
        self.player2_answered = False
        self.player1_answer = None
        self.player2_answer = None
        self.showing_results = False
        self.result_display_time = None
        self.game_over = False
        
    def start_question(self):
        """Start a new question"""
        if self.current_question >= len(self.questions):
            self.game_over = True
            return
        
        self.question_start_time = time.time()
        self.player1_answered = False
        self.player2_answered = False
        self.player1_answer = None
        self.player2_answer = None
        self.showing_results = False
        
    def get_time_remaining(self):
        """Get remaining time for current question"""
        if self.question_start_time is None:
            return self.answer_time
        elapsed = time.time() - self.question_start_time
        remaining = self.answer_time - elapsed
        return max(0, remaining)
    
    def process_answer(self, player, gesture):
        """Process a player's answer gesture - allows changing vote"""
        if self.showing_results or self.game_over:
            return
        
        # Check for thumbs up (True) or thumbs down (False)
        answer = None
        if "Thumb_Up" in gesture or "Thumbs_Up" in gesture:
            answer = True
        elif "Thumb_Down" in gesture or "Thumbs_Down" in gesture:
            answer = False
        
        # Update the current vote (can be changed multiple times)
        if answer is not None:
            if player == 1:
                self.player1_answered = True
                self.player1_answer = answer
            elif player == 2:
                self.player2_answered = True
                self.player2_answer = answer
        else:
            # If no gesture detected, reset the answered status
            if player == 1:
                self.player1_answered = False
                self.player1_answer = None
            elif player == 2:
                self.player2_answered = False
                self.player2_answer = None
    
    def check_answers(self):
        """Check if time is up, then evaluate"""
        if self.showing_results or self.game_over:
            return
        
        time_up = self.get_time_remaining() <= 0
        
        if time_up:
            # Evaluate answers
            correct_answer = self.questions[self.current_question]['answer']
            
            if self.player1_answered and self.player1_answer == correct_answer:
                self.player1_score += POINTS_PER_CORRECT
            
            if self.player2_answered and self.player2_answer == correct_answer:
                self.player2_score += POINTS_PER_CORRECT
            
            # Show results
            self.showing_results = True
            self.result_display_time = time.time()
    
    def next_question(self):
        """Move to next question after showing results"""
        if self.showing_results and time.time() - self.result_display_time >= 3:
            self.current_question += 1
            self.start_question()


def detect_gestures_in_halves(frame, recognizer):
    """Process frame and detect gestures in left and right halves independently"""
    h, w = frame.shape[:2]
    mid_x = w // 2
    
    left_gesture = "None"
    right_gesture = "None"
    
    # Draw vertical line to split screen
    cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
    
    # Process LEFT half
    left_half = frame[:, :mid_x].copy()
    rgb_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
    mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_left)
    left_result = recognizer.recognize(mp_image_left)
    
    if left_result.gestures:
        gesture_list = left_result.gestures[0]
        hand_landmarks = left_result.hand_landmarks[0]
        
        # Draw hand landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * mid_x)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection[0], connection[1]
            start_x = int(hand_landmarks[start_idx].x * mid_x)
            start_y = int(hand_landmarks[start_idx].y * h)
            end_x = int(hand_landmarks[end_idx].x * mid_x)
            end_y = int(hand_landmarks[end_idx].y * h)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        if gesture_list:
            left_gesture = gesture_list[0].category_name
    
    # Process RIGHT half
    right_half = frame[:, mid_x:].copy()
    rgb_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)
    mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_right)
    right_result = recognizer.recognize(mp_image_right)
    
    if right_result.gestures:
        gesture_list = right_result.gestures[0]
        hand_landmarks = right_result.hand_landmarks[0]
        
        # Draw hand landmarks (offset by mid_x)
        for landmark in hand_landmarks:
            x = int(landmark.x * mid_x) + mid_x
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection[0], connection[1]
            start_x = int(hand_landmarks[start_idx].x * mid_x) + mid_x
            start_y = int(hand_landmarks[start_idx].y * h)
            end_x = int(hand_landmarks[end_idx].x * mid_x) + mid_x
            end_y = int(hand_landmarks[end_idx].y * h)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        if gesture_list:
            right_gesture = gesture_list[0].category_name
    
    return frame, left_gesture, right_gesture


def draw_game_ui(frame, game):
    """Draw the game UI on the frame"""
    h, w = frame.shape[:2]
    mid_x = w // 2
    
    if game.game_over:
        # Game over screen
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
        
        cv2.putText(frame, "GAME OVER!", (w//2 - 150, h//2 - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        cv2.putText(frame, f"Player 1 Score: {game.player1_score}", (w//2 - 200, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Player 2 Score: {game.player2_score}", (w//2 - 200, h//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Determine winner
        if game.player1_score > game.player2_score:
            winner = "Player 1 Wins!"
            color = (0, 255, 0)
        elif game.player2_score > game.player1_score:
            winner = "Player 2 Wins!"
            color = (0, 255, 0)
        else:
            winner = "It's a Tie!"
            color = (255, 255, 0)
        
        cv2.putText(frame, winner, (w//2 - 150, h//2 + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        cv2.putText(frame, "Press 'q' to quit", (w//2 - 150, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return
    
    question = game.questions[game.current_question]
    
    # Draw question box at top - make it taller
    cv2.rectangle(frame, (10, 10), (w - 10, 180), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 10), (w - 10, 180), (255, 255, 255), 2)
    
    # Question number and timer
    time_remaining = game.get_time_remaining()
    timer_color = (0, 255, 0) if time_remaining > 3 else (0, 0, 255)
    
    cv2.putText(frame, f"Q {game.current_question + 1}/{len(game.questions)}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {int(time_remaining)}s", 
               (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)
    
    # Question text (word wrap) - improved with smaller font and more lines
    question_text = question['question']
    words = question_text.split()
    lines = []
    current_line = ""
    # Use very small font to ensure it fits
    font_scale = 0.4
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        # Use OpenCV to get actual text size
        (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Leave margins on both sides (40px total)
        if text_width < (w - 40):
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    # If we have too many lines, truncate with "..."
    max_lines = 4
    y_offset = 65
    line_height = 25
    for i, line in enumerate(lines[:max_lines]):
        if i == max_lines - 1 and len(lines) > max_lines:
            # Truncate last line if there are more lines
            line = line[:80] + "..."
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        y_offset += line_height
    
    # Player 1 (left) info
    cv2.putText(frame, "PLAYER 1", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Score: {game.player1_score}", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show current vote during voting period
    if not game.showing_results:
        if game.player1_answered:
            answer_text = "TRUE" if game.player1_answer else "FALSE"
            cv2.putText(frame, f"Vote: {answer_text}", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No Vote Yet", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Thumbs Up = TRUE", (20, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "Thumbs Down = FALSE", (20, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    else:
        # Show final answer after time is up
        if game.player1_answered:
            answer_text = "TRUE" if game.player1_answer else "FALSE"
            cv2.putText(frame, f"Answer: {answer_text}", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Answer", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Player 2 (right) info
    cv2.putText(frame, "PLAYER 2", (mid_x + 20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Score: {game.player2_score}", (mid_x + 20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show current vote during voting period
    if not game.showing_results:
        if game.player2_answered:
            answer_text = "TRUE" if game.player2_answer else "FALSE"
            cv2.putText(frame, f"Vote: {answer_text}", (mid_x + 20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No Vote Yet", (mid_x + 20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Thumbs Up = TRUE", (mid_x + 20, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "Thumbs Down = FALSE", (mid_x + 20, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    else:
        # Show final answer after time is up
        if game.player2_answered:
            answer_text = "TRUE" if game.player2_answer else "FALSE"
            cv2.putText(frame, f"Answer: {answer_text}", (mid_x + 20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Answer", (mid_x + 20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Show results if time is up
    if game.showing_results:
        # Semi-transparent overlay - make it taller for longer explanations
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 250), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        correct_answer = "TRUE" if question['answer'] else "FALSE"
        cv2.putText(frame, f"Correct: {correct_answer}", (w//2 - 150, h - 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Word wrap the explanation to fit inside the overlay
        explanation_text = question['explanation']
        words = explanation_text.split()
        lines = []
        current_line = ""
        font_scale = 0.45
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Leave margins (40px total)
            if text_width < (w - 40):
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Display up to 5 lines of explanation with smaller spacing
        y_offset = h - 170
        line_height = 25
        max_explanation_lines = 5
        for i, line in enumerate(lines[:max_explanation_lines]):
            if i == max_explanation_lines - 1 and len(lines) > max_explanation_lines:
                # Truncate if too long
                line = line[:80] + "..."
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            y_offset += line_height
        
        # Show who got it right
        p1_correct = game.player1_answered and game.player1_answer == question['answer']
        p2_correct = game.player2_answered and game.player2_answer == question['answer']
        
        p1_result = "CORRECT" if p1_correct else "WRONG"
        p2_result = "CORRECT" if p2_correct else "WRONG"
        
        p1_color = (0, 255, 0) if p1_correct else (0, 0, 255)
        p2_color = (0, 255, 0) if p2_correct else (0, 0, 255)
        
        cv2.putText(frame, f"P1: {p1_result}", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, p1_color, 2)
        cv2.putText(frame, f"P2: {p2_result}", (mid_x + 50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, p2_color, 2)


def main():
    """Main game function"""
    print("=" * 80)
    print("Two-Player Quiz Game with Gesture Recognition")
    print("=" * 80)
    
    # Get topic from user
    topic = input("\nEnter the quiz topic: ").strip()
    if not topic:
        print("Error: Topic cannot be empty!")
        return
    
    # Get number of questions
    try:
        num_questions = int(input("Number of questions (default 10): ").strip() or "10")
    except ValueError:
        num_questions = 10
    
    # Get answer time
    try:
        answer_time = int(input("Seconds per question (default 10): ").strip() or "10")
    except ValueError:
        answer_time = 10
    
    print(f"\nGenerating {num_questions} questions about '{topic}'...")
    questions = generate_quiz_questions(topic, num_questions)
    
    if not questions:
        print("Failed to generate questions. Exiting.")
        return
    
    print(f"Successfully generated {len(questions)} questions!")
    print("\nGame Instructions:")
    print("- Player 1 uses LEFT half of screen")
    print("- Player 2 uses RIGHT half of screen")
    print("- Show THUMBS UP for TRUE")
    print("- Show THUMBS DOWN for FALSE")
    print(f"- You have {answer_time} seconds per question")
    print("\nPress ENTER to start...")
    input()
    
    # Initialize game
    game = QuizGame(questions, answer_time)
    cap = cv2.VideoCapture(0)
    
    game.start_question()
    
    print("Game started! Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect gestures
        frame, left_gesture, right_gesture = detect_gestures_in_halves(frame, recognizer)
        
        # Process answers
        game.process_answer(1, left_gesture)
        game.process_answer(2, right_gesture)
        
        # Check if question should end
        game.check_answers()
        
        # Move to next question if showing results
        game.next_question()
        
        # Draw game UI
        draw_game_ui(frame, game)
        
        # Display the frame
        cv2.imshow('Quiz Game - Gesture Recognition', frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 80)
    print("Game Over!")
    print(f"Final Scores:")
    print(f"  Player 1: {game.player1_score}")
    print(f"  Player 2: {game.player2_score}")
    print("=" * 80)


if __name__ == "__main__":
    main()
