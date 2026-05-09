import time
import cv2

# ... (Previous imports and config loading)

def start_assistant():
    # ... (Setup code)
    last_seen_time = time.time()
    standby_mode = False

    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        # Check for hand results
        results = hands.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            last_seen_time = time.time() # Reset the timer
            if standby_mode:
                notify("Vision Assistant", "System Awake")
                standby_mode = False
        else:
            # If no hand for 5 seconds, go to standby
            if time.time() - last_seen_time > 5 and not standby_mode:
                notify("Vision Assistant", "Entering Standby Mode")
                standby_mode = True

        if standby_mode:
            # Apply a "Dimming" effect to the display to show it's asleep
            image = cv2.addWeighted(image, 0.2, image, 0, 0)
            cv2.putText(image, "STANDBY - NO HAND DETECTED", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # ... (Rest of your gesture logic)