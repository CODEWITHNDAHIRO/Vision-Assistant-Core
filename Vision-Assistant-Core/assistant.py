import cv2
import mediapipe as mp

def start_assistant():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)

    # Use 'with' to handle the AI model memory safely
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        print("AI Assistant (v2.0) is Active...")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            # Draw the detections
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    print("Status: User Present")

            cv2.imshow('Vision Assistant Core', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_assistant()
