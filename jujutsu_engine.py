import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm

class jujutsu_engine:
    def __init__(self, name, video_path):
        self.name = name
        self.video_path = video_path
        self.stream = cv2.VideoCapture(video_path)   

    WRIST = 0

    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def check_gesture(self , lmlist):
        #Must be overridden by subclasses to define gesture math.
        raise NotImplementedError("Each technique must implement its own gesture criteria.")  

       
    def landmark_distance(self , lmlist , point1 , point2):
         return math.hypot(lmlist[point1][1] - lmlist[point2][1],lmlist[point1][2] - lmlist[point2][2])


    def get_hand_scale(self, lmlist):
        scale = self.landmark_distance(lmlist,self.WRIST,self.MIDDLE_MCP)
        return max(scale, 1)

    
    def get_frame(self):
        #Safely reads and loops the internal video stream asset.
        ret , frame  = self.stream.read()
        if not ret:
            total = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(self.stream.get(cv2.CAP_PROP_FPS))
            loop_frame = int(self.loop_sec * fps)
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, total - loop_frame)
            #self.intro_finished = True
            ret, frame = self.stream.read()
               
        return cv2.resize(frame, (1280, 720))
    
    def reset_video(self):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self):
        self.stream.release()

class InfiniteVoid(jujutsu_engine):
        def __init__(self, video_path):
            super().__init__("void", video_path)
            self.loop_sec = 0.15
            self.exit_delay = 5
        def check_gesture(self, lmlist):
                hand_scale = self.get_hand_scale(lmlist)
                
                dist_index =self.landmark_distance(lmlist,self.WRIST,self.INDEX_TIP)
                dist_middle = self.landmark_distance(lmlist,self.WRIST,self.MIDDLE_TIP)
                dist_ring = self.landmark_distance(lmlist,self.WRIST,self.RING_TIP)
                dist_pinky = self.landmark_distance(lmlist,self.WRIST,self.PINKY_TIP)

                tip_cross_dist =self.landmark_distance(lmlist,self.INDEX_TIP,self.MIDDLE_TIP)

                is_index_extended = dist_index > (hand_scale * 1.20)
                is_middle_extended = dist_middle > (hand_scale * 1.20)
                
                is_ring_folded = dist_ring < (hand_scale * 1.20)
                is_pinky_folded = dist_pinky < (hand_scale * 1.20)
                
                is_crossed = tip_cross_dist < (hand_scale * 0.90)

                return (is_index_extended and is_middle_extended and 
                     is_ring_folded and is_pinky_folded and is_crossed)

class ReversalRed(jujutsu_engine):
    def __init__(self , video_path):
         super().__init__("red",video_path)
         self.loop_sec = 0.30
         self.exit_delay = 4

    def check_gesture(self , lmlist):
            hand_scale = self.get_hand_scale(lmlist)
            #dist_thumb = math.hypot(lmlist[4][1] - wrist_x, lmlist[4][2] - wrist_y)    
            dist_index = self.landmark_distance(lmlist,self.WRIST,self.INDEX_TIP)
            dist_middle = self.landmark_distance(lmlist,self.WRIST,self.MIDDLE_TIP)
            dist_ring = self.landmark_distance(lmlist,self.WRIST,self.RING_TIP)
            dist_pinky = self.landmark_distance(lmlist,self.WRIST,self.PINKY_TIP)

            #is_thumb_folded = dist_thumb < (hand_scale * 1.00)
            is_index_extended = (lmlist[self.INDEX_TIP][2] < lmlist[self.INDEX_DIP][2] < lmlist[self.INDEX_PIP][2] < lmlist[self.INDEX_MCP][2]) and (dist_index > (hand_scale * 1.30))
            is_middle_folded = dist_middle < (hand_scale * 1.00)         
            is_ring_folded = dist_ring < (hand_scale * 1.00)
            is_pinky_folded = dist_pinky < (hand_scale * 1.00) 

            return ( is_index_extended and is_middle_folded and 
                     is_ring_folded and is_pinky_folded)
  
class LapseBlue(jujutsu_engine):
        def __init__(self , video_path):
            super().__init__("blue",video_path)
            self.loop_sec = 0.30
            self.exit_delay = 4
        def check_gesture(self , lmlist):
            hand_scale = self.get_hand_scale(lmlist)   
            dist_thumb = self.landmark_distance(lmlist , self.WRIST , self.THUMB_TIP)
            dist_index = self.landmark_distance(lmlist,self.WRIST,self.INDEX_TIP)
            dist_middle = self.landmark_distance(lmlist,self.WRIST,self.MIDDLE_TIP)
            dist_ring = self.landmark_distance(lmlist,self.WRIST,self.RING_TIP)
            dist_pinky = self.landmark_distance(lmlist,self.WRIST,self.PINKY_TIP)

            is_thumb_extended = dist_thumb > hand_scale * 1.0
            is_index_extended = ( lmlist[self.INDEX_TIP][2] < lmlist[self.INDEX_DIP][2] < lmlist[self.INDEX_PIP][2] < lmlist[self.INDEX_MCP][2]) and (dist_index > hand_scale * 1.30)
            is_middle_extended = ( lmlist[self.MIDDLE_TIP][2] < lmlist[self.MIDDLE_DIP][2] < lmlist[self.MIDDLE_PIP][2] < lmlist[self.MIDDLE_MCP][2]) and (dist_middle > hand_scale * 1.30)        
            is_ring_extended = ( lmlist[self.RING_TIP][2] < lmlist[self.RING_DIP][2] < lmlist[self.RING_PIP][2] < lmlist[self.RING_MCP][2]) and (dist_ring > hand_scale * 1.30)
            is_pinky_extended = ( lmlist[self.PINKY_TIP][2] < lmlist[self.PINKY_DIP][2] < lmlist[self.PINKY_PIP][2] < lmlist[self.PINKY_MCP][2]) and (dist_pinky > hand_scale * 1.30)


            #**CHECKING SPREAD**
            thumb_index = self.landmark_distance(lmlist , self.THUMB_TIP , self.INDEX_TIP)
            index_middle = self.landmark_distance(lmlist , self.INDEX_TIP , self.MIDDLE_TIP)
            middle_ring = self.landmark_distance(lmlist , self.MIDDLE_TIP , self.RING_TIP)
            ring_pinky = self.landmark_distance(lmlist , self.RING_TIP , self.PINKY_TIP)


            # print(
            #     f"T-I : {thumb_index / hand_scale:.2f}",
            #     f"I-M : {index_middle / hand_scale:.2f}",
            #     f"M-R : {middle_ring / hand_scale:.2f}",
            #     f"R-P : {ring_pinky / hand_scale:.2f}"
            # )
            are_spread = (
            thumb_index > hand_scale * 1.0 and
            index_middle > hand_scale * 0.45 and
            middle_ring > hand_scale * 0.35 and
            ring_pinky > hand_scale * 0.50
        )

            return ( is_thumb_extended and is_index_extended 
                    and is_middle_extended and is_ring_extended and is_pinky_extended and are_spread
            )
        
class JujutsuEngine:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.detector = htm.HandDetector()
        
        self.background = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.height_bg, self.width_bg = self.background.shape[:2]
        
        self.alpha = 0.0
        self.transition_rate = 0.06
        self.exit_time = None          
        self.should_exit = False
        self.techniques = []
        self.active_technique = None
        self.previous_technique = None
        self.is_switching = False
        self.switch_alpha = 0.0
        self.switch_rate = 0.08

    def register_technique(self, technique):
        self.techniques.append(technique)

    def update_states(self, lmlist):
        gesture_detected = False 

        if len(lmlist) != 0:
            for technique in self.techniques:
                if technique.check_gesture(lmlist):
                    gesture_detected = True
                    if self.active_technique != technique:
                        if self.active_technique is not None:
                            self.active_technique.reset_video()

                        self.active_technique = technique
                        if hasattr(self.active_technique, "reset"):
                            self.active_technique.reset()
                        else:
                            self.active_technique.reset_video()
                        self.previous_technique = None
                        self.is_switching = False
                        self.switch_alpha = 0.0
                        self.alpha = 0.0
                    break

        if gesture_detected:
            self.should_exit = False
            self.exit_time = None
            self.alpha = min(1.0, self.alpha + self.transition_rate)
        elif self.alpha > 0:
            if self.exit_time is None:
                self.exit_time = time.time()

            if (self.active_technique is not None and time.time() - self.exit_time >= self.active_technique.exit_delay):
                self.should_exit = True
    def build_canvas(self, img):
        frame = self.background.copy()
   

        if self.alpha > 0.0 and self.active_technique:

            domain_frame = self.active_technique.get_frame()

            frame = cv2.addWeighted(self.background,1.0 - self.alpha,domain_frame,self.alpha,0)

        cap_resized = cv2.resize(img, (280, 265))
        height_cap, width_cap = cap_resized.shape[:2]
        y = self.height_bg - height_cap     
        frame[y:y+height_cap, 0:width_cap] = cap_resized    

        return frame  
    def start(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            img = cv2.resize(img, (1280, 720))
            img = self.detector.findHands(img)
            lmlist = self.detector.findPosition(img)

            if len(lmlist) == 0:
                print("⚠️ Camera Status: No hand visible. (Increase room lighting!)", end="\r")
            else:
                print(f"✅ Camera Status: Tracking active ({len(lmlist)} points detected)", end="\r")

            self.update_states(lmlist)
            if self.should_exit:
                break
            output_canvas = self.build_canvas(img)
            
            cv2.imshow("video", output_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        for technique in  self.techniques:
            technique.release()
        cv2.destroyAllWindows()