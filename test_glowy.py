import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh()
mpDraw = mp.solutions.drawing_utils
mpDraw_styles = mp.solutions.drawing_styles
pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face_landmarks, mpFace.FACEMESH_TESSELATION,
                    connection_drawing_spec=mpDraw_styles.get_default_face_mesh_tesselation_style())
            for id, lm in enumerate(face_landmarks.landmark):
                print(id, lm)
    
   
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, 
                (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)