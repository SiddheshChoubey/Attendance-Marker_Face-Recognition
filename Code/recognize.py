import datetime
import os
import time
import cv2
import pandas as pd


def recognize_attendence():
    # Create directories if they don't exist
    os.makedirs("StudentDetails", exist_ok=True)
    os.makedirs("Attendance", exist_ok=True)
    
    # Create or verify StudentDetails.csv
    student_details_path = os.path.join("StudentDetails", "StudentDetails.csv")
    if not os.path.exists(student_details_path):
        # Create new CSV with headers
        df = pd.DataFrame(columns=['Id', 'Name'])
        df.to_csv(student_details_path, index=False)
        print("Created new StudentDetails.csv file")
        print("Please add student data first using Capture Faces option")
        return
    
    # Load and verify the CSV file
    try:
        df = pd.read_csv(student_details_path)
        if 'Id' not in df.columns or 'Name' not in df.columns:
            print("Invalid CSV format. Recreating the file...")
            df = pd.DataFrame(columns=['Id', 'Name'])
            df.to_csv(student_details_path, index=False)
            print("Please add student data first using Capture Faces option")
            return
        
        if len(df) == 0:
            print("No student data found. Please add students first using Capture Faces option")
            return
            
    except Exception as e:
        print(f"Error reading StudentDetails.csv: {e}")
        return
    
    # Verify cascade file exists and use absolute path
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")
    
    faceCascade = cv2.CascadeClassifier(cascade_path)
    cam = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load the trainer
    trainer_path = "TrainingImageLabel/Trainner.yml"
    if not os.path.exists(trainer_path):
        raise FileNotFoundError("Trainer file not found. Please train the system first.")
    recognizer.read(trainer_path)
    
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    # start realtime video capture
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640) 
    cam.set(4, 480) 
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,
                minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 100:
                try:
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    if len(aa) == 0:
                        tt = f"{Id}-Unknown"
                    else:
                        tt = f"{Id}-{aa[0]}"
                except:
                    tt = f"{Id}-Error"
                confstr = "  {0}%".format(round(100 - conf))
            else:
                Id = '  Unknown  '
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))

            if (100-conf) > 67:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = str(aa)[2:-2]
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            tt = str(tt)[2:-2]
            if(100-conf) > 67:
                # Changed from "[Pass]" to show ID and Name
                display_text = f"ID: {Id} - {aa}"
                cv2.putText(im, display_text, (x+5,y-5), font, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100-conf) > 67:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
            elif (100-conf) > 50:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)


        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()