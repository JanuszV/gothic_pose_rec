import cv2
import mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2


class handDetector():
    def __init__(self, mode = False, num_hands = 2, model_complexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode 
        self.num_hands = num_hands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.num_hands, self.model_complexity,  self.detectionCon, self.trackCon)

    def findHand(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hand_conns = 0
        if self.results.multi_hand_landmarks:
            if draw:
                for handLM in self.results.multi_hand_landmarks:
                    hand_conns = self.mpHands.HAND_CONNECTIONS
                    self.mpDraw.draw_landmarks(img, handLM, hand_conns)
        
        return hand_conns
    
    def getPosition(self, img):
        lmList = []
        if self.results.multi_hand_landmarks:
            for id, handLms in enumerate(self.results.multi_hand_landmarks):
                lmList_hand = []
                for id, lm in enumerate(handLms.landmark):
                    h , w , c = img.shape
                    cx, cy, cz = int(lm.x *w), int(lm.y *h), int(lm.z * w)
                    lmList_hand.append([id,cx,cy, cz])
                lmList.append(lmList_hand)
        return lmList
        


class poseDetector():

    def __init__(self, mode = False, model_complexity = 1, smooth = True, seg = False, detectionCon = 0.5, trackCon = 0.5, without_hands_and_head = True):

        self.mode = mode
        self.model_complexity = model_complexity
        self.seg = seg
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        if without_hands_and_head:
            self.ignore_id = list(range(11)) + list(range(17,23)) #not using face and hands from pose module
            self.pose_conn = frozenset([
                              (0,1), (1,3), (3,5), (0,2), (2,4),(0,6), (1,7), (6,7)])
        else:
            self.ignore_id = []
            self.pose_conn = self.mpPose.POSE_CONNECTIONS
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth, self.seg, self.detectionCon, self.trackCon)
        
    
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filtered_data = []
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                    if id not in self.ignore_id:
                        filtered_data.append(landmark)
                self.filtered_data_land = landmark_pb2.NormalizedLandmarkList(landmark = filtered_data)   
                self.mpDraw.draw_landmarks(img, self.filtered_data_land, self.pose_conn) 
        return self.pose_conn

    def getPosition(self, img):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.filtered_data_land.landmark):
                h , w , c = img.shape
                cx, cy, cz = int(lm.x *w), int(lm.y *h), int(lm.z * w)
                lmList.append([id,cx,cy, cz])
        return lmList

class faceDetector():
    def __init__(self, mode = False, num_faces = 1, refine_landmarks = False, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode 
        self.num_faces = num_faces
        self.refine_landmarks = refine_landmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFace = mp.solutions.face_mesh
        self.mpDraw_styles = mp.solutions.drawing_styles
        self.face = self.mpFace.FaceMesh(self.mode,self.num_faces, self.refine_landmarks,  self.detectionCon, self.trackCon)

    def findFace(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        face_conns = 0
        if self.results.multi_face_landmarks:
            if draw:
                for face_landmarks in self.results.multi_face_landmarks:
                    face_conns = self.mpFace.FACEMESH_TESSELATION
                    self.mpDraw.draw_landmarks(img, face_landmarks, face_conns,
                           )
        return face_conns
    def getPosition(self, img):
        lmList = []
        if self.results.multi_face_landmarks:
            for face_landmark in self.results.multi_face_landmarks:
                for id, lm in enumerate(face_landmark.landmark):
                    h , w , c = img.shape
                    cx, cy, cz = int(lm.x *w), int(lm.y *h), int(lm.z * w)
                    lmList.append([id,cx,cy, cz])
        return lmList
    
#implementation 
def main(cap = cv2.VideoCapture(0)):
    pTime = 0
    cTime = 0
    pose_detector = poseDetector()
    hand_detector = handDetector()
    face_detector = faceDetector()
    print_once = False
    while True:
        success, img = cap.read()
        cTime = time.time()
        
        
        fps = 1/(cTime - pTime)
        pTime = cTime
        pose_conns = pose_detector.findPose(img)
        hand_conns = hand_detector.findHand(img)
        face_conns = face_detector.findFace(img)
        pose_position = pose_detector.getPosition(img)
        hand_position = hand_detector.getPosition(img)
        face_position = face_detector.getPosition(img)
        cv2.putText(img, str(int(fps)), (70,30), cv2.FONT_HERSHEY_PLAIN , 3 , (255, 0,0), 3)
        cv2.imshow("Kamera na Beziego", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()