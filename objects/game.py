#Code from @LynnHaDo on github, modified by Amelia Henzel

from . import Drop 

# Import modules for rendering video
import cv2
import time 
import numpy as np
import math 

# Import mediapipe modules
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing 
import mediapipe.python.solutions.drawing_styles as drawing_styles

# Set up the model
model = mp_hands.Hands(
    static_image_mode = False, 
    max_num_hands=1, 
    min_detection_confidence=0.6
)

#@Author AMELIA moved to bottom of screen
Y_THRESHOLD = 450 # everything moved below this y will be 

class Game:
    """
    Represents a game object with several functionalities
    """
    def __init__(self, title):
        self.title = title # title of the game

        self.geese = [] # list of geese currently in the game
        
        self.lives = 20 # say we allow 20 lives at the start of the game
        self.score = 0 
        self.speed = [0, 5] # the speed by which objects are moving
        self.drop_rate = 1 # rate by which the geese are generated
        self.difficulty_level = 1 
        self.game_over = False 

        self.curFrame = 0
        self.prevFrame = 0 
        self.next_drop_time = 0 # when the next drop will come. 

        # Index finger movement 
        self.index_movement = np.array([[]], np.int32)
        self.index_movement_length = 19
    
    def create_drop(self):
        """
        Create a new drop
        """
        #@Author AMELIA raindrops spawn at top of screen
        drop = Drop(self.xLimit, (-self.yLimit))
        self.geese.append(drop)
    
    def move_geese(self, img):
        """
        Move the geese created
        """
        # Make a copy
        # We don't want to remove while iterating over a list!
        geese_copy = self.geese.copy()

        for drop in geese_copy: 
            if drop.is_passed_border(Y_THRESHOLD):
                self.lives -= 1 
                self.geese.remove(drop)
            
            y_offset, y_end, x_offset, x_end  = drop.get_position() 

            # This messy code is blending the image of the drop onto the background image

            if y_end <= img.shape[0] and x_end <= img.shape[1]:
                b, g, r, a = cv2.split(drop.image) 
                a = a/255.0 
                img1_bgr = cv2.merge([b*a, g*a, r*a]) # multiply all channels of `img1` with the alpha channel
                
                img2_roi = img[y_offset:y_end, x_offset:x_end,:]
                img2_roi_bgr = cv2.split(img2_roi)
                
                if (len(img2_roi_bgr) == 0):
                    return 
                
                bg_b, bg_g, bg_r = img2_roi_bgr

                # Merge them with opposite alpha levels
                img2_roi = cv2.merge([bg_b * (1 - a), bg_g * (1 - a), bg_r * (1 - a)])

                cv2.add(img1_bgr, img2_roi, img2_roi) # Add img1
                # Replace the portion of the original img2 with the new blended image

                img[y_offset:y_end, x_offset:x_end,:] = img2_roi

            drop.set_next_position(self.speed)
    
    def distance(self, obj1: tuple[int, int], obj2: tuple[int, int]) -> int:
        """
        Helper to calculate the distance between 2 objects
        """
        x1 = obj1[0]
        x2 = obj2[0]

        y1 = obj1[1]
        y2 = obj2[1]

        #@Author AMELIA added y1 and y2 for negative values
        d = math.sqrt(pow(x1 - x2, 2) + pow(y1 + y2, 2))
        return int(d)
    
    def start(self):
        """
        Initiate the game. 
        
        * Start the video camera 
        * Stop when `q` is pressed on the keyboard. 

        """
        # Start the video
        cap = cv2.VideoCapture(0)

        while (cap.isOpened()):
            self.xLimit = int(cap.get(3)) # get width
            self.yLimit = int(cap.get(4)) # get height

            success, img = cap.read() # read a frame

            if not success:
                continue # skip to the next frame 
                
            h, w, c = img.shape
            # Convert to RGB
            img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            # Detect the results
            result = model.process(img)
            # Converting back the RGB image to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = np.ones((h, w, 3), dtype=np.uint8) * 255

            # Draw the divider 
            cv2.line(img, (0, Y_THRESHOLD), (w, Y_THRESHOLD), (255, 0, 0), 3)

            # Handle events when the hand is captured on the screen
            if result.multi_hand_landmarks:
                hand_landmarks_list = result.multi_hand_landmarks
                # Loop through 21 landmarks
                for hand_landmarks in hand_landmarks_list:
                    # Visualize the landmarks
                    mp_drawing.draw_landmarks(
                        img, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS, 
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style()
                    )

                    # Catch the movement of the index finger
                    if hand_landmarks.landmark:
                        index_landmark = hand_landmarks.landmark[8] # It's the 8th item in the result
                        # Get the position
                        index_pos = (int(index_landmark.x * w), int(index_landmark.y * h))
                        # Draw a circle at the index
                        cv2.circle(img, index_pos, 18, (255, 0, 0), -1)

                        # This is basically drawing the movement of the index finger 
                        self.index_movement = np.append(self.index_movement, index_pos)

                        while len(self.index_movement) >= self.index_movement_length:
                            self.index_movement = np.delete(self.index_movement, 
                                                            len(self.index_movement) - self.index_movement_length, 
                                                            0)

                        geese_copy = self.geese.copy()

                        for drop in geese_copy:
                            
                            d = self.distance(drop.curPos, index_pos)
                        
                            if (d < drop.size + 100):
                                self.score += 100 
                                self.geese.remove(drop)

            # Unlock a new level
            if self.score > 0 and self.score % 1000 == 0: 
                self.difficulty_level = int(self.score / 1000) + 1
                self.drop_rate = self.difficulty_level * 0.5
                self.speed[1] = int(2.5 * self.difficulty_level)
            
            if self.lives <= 0: 
                self.game_over = True 

            self.index_movement = self.index_movement.reshape((-1, 1, 2))
            # draw the movement of the index finger
            cv2.polylines(img, [self.index_movement], False, (0, 0, 0), 15, 0) 

            self.curFrame = time.time() 

            # display score metrics
            cv2.putText(img, "Score: " + str(self.score), (int(w * 0.1), 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)
            cv2.putText(img, "Level: " + str(self.difficulty_level), (int(w * 0.7), 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)
            cv2.putText(img, "Lives: " + str(self.lives), (int(w * 0.4), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 0), 2)

            self.prevFrame = self.curFrame

            if self.game_over:
                cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (28, 49, 235), 13)
                self.geese.clear()
            else: 
                if (time.time() > self.next_drop_time):
                    self.create_drop() 
                    self.next_drop_time = time.time() + 1/self.drop_rate
                self.move_geese(img)

            cv2.imshow(self.title, img)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()    