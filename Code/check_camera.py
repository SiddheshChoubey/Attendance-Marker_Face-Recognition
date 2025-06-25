import cv2
import os

def camer():
    # Verify cascade file exists and use absolute path
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")
    
    cascade_face = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_face.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        cv2.imshow('FACE DETECTION', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
