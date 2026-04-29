import cv2
import time

def start_assistant():
    #using the default webcam
    video = cv2.videoCapture(0)
    
    print("AI assistant is waking up...")
    while True:
        check, frame = video.read()
        if not check:
            break
        #basic grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #we will use a basic face detector for day1
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "user", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imshow("AI Assistant view", frame)
            # press 'q' to shutdown
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()
    if __name__ == "__main__":
        start_assistant()