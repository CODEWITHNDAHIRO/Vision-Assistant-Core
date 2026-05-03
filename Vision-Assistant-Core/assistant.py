import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import math
import os

def set_volume(percentage):
    # This sends a command to  Mac to set the volume
    # It maps 0-100 to the system's 0-7 range
    volume_level = int(percentage / 14) 
    os.system(f"osascript -e 'set volume {volume_level}'")

def start_assistant():
    
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        print("Vision Assistant (v5.0) - Audio Integration Active")

        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]

                    # Calculate distance
                    dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
                    
                    # Mapping: Pinch (close) = 0%, Wide (open) = 100%
                    vol = int(dist * 400) # Adjusted multiplier for better sensitivity
                    if vol > 100: vol = 100
                    
                    # Actively change volume if hand is detected
                    set_volume(vol)
                    
                    cv2.putText(image, f"System Volume: {vol}%", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Vision Assistant Core', image)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_assistant()
