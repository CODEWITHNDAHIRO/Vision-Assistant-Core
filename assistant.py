import cv2
import mediapipe as mp
import math
import os

def set_volume(percentage):
    volume_level = int(percentage / 14) 
    os.system(f"osascript -e 'set volume {volume_level}'")

def start_assistant():
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    is_active = False

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        frame_count = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            frame_count += 1
            #this only runs the AI every 3rd frame to save CPU
            if frame_count % 3 != 0:
                cv2.imshow('Vision Assistant Core v2.0-Alpha', image)
                if cv2.waitKey(5) & 0xFF == ord('q'): break
                continue
            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    
                    # Peace Sign Toggle Logic
                    if lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y > lm[14].y:
                        is_active = not is_active
                        cv2.waitKey(400)

                    if is_active:
                        dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
                        vol = min(int(dist * 400), 100)
                        set_volume(vol)
                        
                        # UI: Volume Bar
                        cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                        cv2.rectangle(image, (50, int(400 - (vol*2.5))), (85, 400), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, f"{vol}%", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # UI: Status Indicator
            color = (0, 255, 0) if is_active else (0, 0, 255)
            status = "SYSTEM: ACTIVE" if is_active else "SYSTEM: IDLE"
            cv2.putText(image, status, (w-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Vision Assistant Core v1.0', image)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_assistant()