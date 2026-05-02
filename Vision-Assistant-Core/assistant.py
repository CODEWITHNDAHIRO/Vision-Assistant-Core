import cv2
import mediapipe as mp
import math

def start_assistant():
    mp_hands = mp.solutions.hands
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        print("Vision Assistant (v4.0) - Gesture Engine Online")

        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get Thumb (4) and Index (8) coordinates
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]

                    # Calculate distance
                    dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
                    
                    # Convert distance to a readable "Volume" percentage
                    # We will map 0.02 - 0.2 distance to 0 - 100%
                    vol = int(dist * 500) 
                    
                    # Visual feedback: Draw a line between fingers
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(image, f"Control: {vol}%", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Vision Assistant Core', image)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_assistant()
