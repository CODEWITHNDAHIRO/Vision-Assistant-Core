import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import math
import os

def set_volume(percentage):
    volume_level = int(percentage / 14) 
    os.system(f"osascript -e 'set volume {volume_level}'")

def start_assistant():
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    
    # State variable: Is the volume control currently active?
    is_active = False

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        print("Vision Assistant (v6.0) - Interaction Logic Active")

        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    
                    # 1. Detection Logic for Peace Sign (Index and Middle Up)
                    # Checking if fingertips (8, 12) are higher than middle joints (6, 10)
                    index_up = landmarks[8].y < landmarks[6].y
                    middle_up = landmarks[12].y < landmarks[10].y
                    ring_down = landmarks[16].y > landmarks[14].y
                    
                    if index_up and middle_up and ring_down:
                        is_active = not is_active # Toggle state
                        print(f"Volume Control: {'ENABLED' if is_active else 'DISABLED'}")
                        cv2.waitKey(500) # Simple debounce to prevent rapid flickering

                    # 2. Volume Logic (Only runs if is_active is True)
                    if is_active:
                        dist = math.hypot(landmarks[8].x - landmarks[4].x, landmarks[8].y - landmarks[4].y)
                        vol = min(int(dist * 400), 100)
                        set_volume(vol)
                        cv2.putText(image, f"ACTIVE: {vol}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "IDLE: Show Peace Sign", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Vision Assistant Core', image)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_assistant()
