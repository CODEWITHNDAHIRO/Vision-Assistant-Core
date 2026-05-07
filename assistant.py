import cv2
import mediapipe as mp
import os

def notify(title, text):
    # Triggers a native macOS notification
    os.system(f"""
              osascript -e 'display notification "{text}" with title "{title}"'
              """)

def start_assistant():
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    is_active = False
    
    with mp_hands.Hands(min_detection_confidence=0.8) as hands:
        print("Vision Assistant v2.0 - Background Engine Running...")

        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            # (AI processing logic remains the same)
            results = hands.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    
                    # Peace Sign Toggle
                    if lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y > lm[14].y:
                        is_active = not is_active
                        
                        # NEW: Notify the user of the state change
                        status_msg = "Gesture Control ON" if is_active else "Gesture Control OFF"
                        notify("Vision Assistant", status_msg)
                        
                        cv2.waitKey(600) # Prevents double-toggling

            # HEADLESS MODE: In a real background app, we eventually remove cv2.imshow
            # For today, keep it, but try running it without the window later!
            cv2.imshow('Vision Assistant', image)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()