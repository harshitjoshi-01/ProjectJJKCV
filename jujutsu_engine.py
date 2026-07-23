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

    def check_gesture(self , lmlist):
        #Must be overridden by subclasses to define gesture math.
        raise NotImplementedError("Each technique must implement its own gesture criteria.")     
    
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
                wrist = lmlist[0]
                m_knuckle = lmlist[9] 
        
                wrist_x, wrist_y = wrist[1], wrist[2]
                knuckle_x, knuckle_y = m_knuckle[1], m_knuckle[2]
        
                hand_scale = math.hypot(knuckle_x - wrist_x, knuckle_y - wrist_y)
                if hand_scale == 0:
                    hand_scale = 1
                
                dist_index = math.hypot(lmlist[8][1] - wrist_x, lmlist[8][2] - wrist_y)
                dist_middle = math.hypot(lmlist[12][1] - wrist_x, lmlist[12][2] - wrist_y)
                dist_ring = math.hypot(lmlist[16][1] - wrist_x, lmlist[16][2] - wrist_y)
                dist_pinky = math.hypot(lmlist[20][1] - wrist_x, lmlist[20][2] - wrist_y)

                tip_cross_dist = math.hypot(lmlist[8][1] - lmlist[12][1], lmlist[8][2] - lmlist[12][2])

                is_index_extended = dist_index > (hand_scale * 1.2)
                is_middle_extended = dist_middle > (hand_scale * 1.2)
                
                is_ring_folded = dist_ring < (hand_scale * 1.20)
                is_pinky_folded = dist_pinky < (hand_scale * 1.20)
                
                is_crossed = tip_cross_dist < (hand_scale * 0.90)

                return (is_index_extended and is_middle_extended and 
                     is_ring_folded and is_pinky_folded and is_crossed)
                
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
                        self.previous_technique = self.active_technique
                        self.active_technique = technique
                        self.active_technique.reset_video()
                        self.alpha = 0.0
                        # if self.previous_technique is None:
                        #     self.alpha = 0.0
                        self.is_switching = True
                        self.switch_alpha = 0.0
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
        if self.is_switching:

            self.switch_alpha += self.switch_rate

            if self.switch_alpha >= 1.0:
                self.switch_alpha = 1.0
                self.is_switching = False
                if self.previous_technique:
                    self.previous_technique.reset_video()
                self.previous_technique = None

    def build_canvas(self, img):
        frame = self.background.copy()
   

        if self.alpha > 0.0 and self.active_technique:

            if self.is_switching and self.previous_technique:

                old_frame = self.previous_technique.get_frame()
                new_frame = self.active_technique.get_frame()

                domain_frame = cv2.addWeighted(old_frame,1.0 - self.switch_alpha,new_frame,self.switch_alpha,0)

            else:
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