import time
import cv2
import mediapipe as mp
import subprocess

def notify(title, message):
    subprocess.run(["osascript", "-e", 
        f'display notification "{message}" with title "{title}"'])

def start_assistant():
    last_seen_time = time.time()
    standby_mode = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        results = hands.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            last_seen_time = time.time()
            if standby_mode:
                notify("Vision Assistant", "System Awake")
                standby_mode = False
        else:
            if time.time() - last_seen_time > 5 and not standby_mode:
                notify("Vision Assistant", "Entering Standby Mode")
                standby_mode = True

        if standby_mode:
            image = cv2.addWeighted(image, 0.2, image, 0, 0)
            cv2.putText(image, "STANDBY - NO HAND DETECTED", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Vision Assistant", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_assistant()