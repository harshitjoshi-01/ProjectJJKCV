import  cv2
import mediapipe as mp
import time

class HandDetector():

    
    def __init__(self, mode = False , max_num_hands = 2 ,model_complexity = 0 , min_detection_confidence = 0.5 , min_tracking_confidence = 0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode ,self.max_num_hands, self.model_complexity , self.min_detection_confidence , self.min_tracking_confidence
 )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self , img , draw = True):
     
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
            #print(result.multi_hand_landmarks)

        if (self.result.multi_hand_landmarks):
                for handlms in self.result.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
                    
        return img


    def findPosition(self , img , HandNo=0):
            lmlist = []

            if hasattr(self, 'result') and self.result and self.result.multi_hand_landmarks:
                myHand = self.result.multi_hand_landmarks[HandNo]    
                for id , lms in  enumerate(myHand.landmark):
                    #print(id,lms)
                    h,w,c = img.shape
                    cx , cy , cz = int(lms.x*w) , int(lms.y*h) , int(lms.z*w)
                    #print(id , cx ,cy)
                    lmlist.append([id , cx , cy , cz])

            return lmlist


def main():
    Ctime=0
    Ptime=0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success , img = cap.read()
        if not success:
            print("Failed to capture image")
            break    
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[12])
        
        Ctime=time.time()
        fps = 1/(Ctime-Ptime)
        Ptime = Ctime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),3)

        cv2.imshow("Livevideo", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()   
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()