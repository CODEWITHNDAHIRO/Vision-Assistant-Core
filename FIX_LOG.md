# Vision-Assistant-Core — Bug Fix Log

## Date: May 9, 2026

-----

## Issues Found & Fixed

### 1. Missing `cv2.VideoCapture()` initialization

**Problem:** The `cap` variable was never defined, so `while cap.isOpened()` crashed immediately with a `NameError`.  
**Fix:** Added `cap = cv2.VideoCapture(0)` at the start of `start_assistant()`.

-----

### 2. Nested/duplicate `start_assistant()` function

**Problem:** A second `def start_assistant():` was accidentally defined inside the first one. The outer function would just redefine the inner one and exit — the camera code never ran.  
**Fix:** Removed the duplicate inner function definition.

-----

### 3. Duplicate `while cap.isOpened()` loop

**Problem:** Two `while cap.isOpened()` loops existed. The first one contained only `...` (a placeholder), causing an infinite loop that prevented the real loop from ever running.  
**Fix:** Removed the placeholder loop, keeping only the complete one.

-----

### 4. `cap.release()` inside the loop

**Problem:** `cap.release()` was placed inside the `while` loop, releasing the camera on every frame.  
**Fix:** Moved `cap.release()` and `cv2.destroyAllWindows()` to after the loop ends.

-----

### 5. `notify()` function not defined

**Problem:** `notify("Vision Assistant", "...")` was called but never defined, causing a `NameError` crash every time a hand was detected or standby mode triggered.  
**Fix:** Defined a `notify()` helper using macOS `osascript` via `subprocess`:

```python
import subprocess

def notify(title, message):
    subprocess.run(["osascript", "-e",
        f'display notification "{message}" with title "{title}"'])
```

-----

### 6. macOS Camera Permissions

**Problem:** Terminal was not authorized to access the camera. OpenCV returned `Camera opened: False` and crashed on `cv2.imshow()`.  
**Fix:** Granted camera access to Terminal via:  
**System Settings → Privacy & Security → Camera → Enable Terminal**

-----

### 7. Missing `cv2.imshow()` call

**Problem:** The camera was running but no window was ever displayed because `cv2.imshow()` was not in the loop.  
**Fix:** Added inside the loop:

```python
cv2.imshow("Vision Assistant", image)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

-----

## Final Working Structure

```python
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
```

-----

## How to Run

```bash
conda activate vision-env
cd ~/Documents/Vision-Assistant-Core
python assistant.py
```

Press `q` to quit the window.