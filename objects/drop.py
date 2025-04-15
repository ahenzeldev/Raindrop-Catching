#Code modified by Amelia Henzel @LynnHaDo
import random 
import cv2 
import os

class Drop:
    """
    Represents a drop to catch
    """
    def __init__(self, xLimit: int, yLimit: int):
        IMAGE_PATH = os.path.join(os.getcwd(), 'objects', 'assets', 'drop.png')
        hd_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        
        self.curPos = [random.randint(15, xLimit - 15), yLimit]
        self.nextPos = [0, 0]

        self.size = int(xLimit * 0.1)
        self.image = cv2.resize(hd_image, (self.size, self.size))
    
    #@Author AMELIA changed to greater than -70
    def is_passed_border(self, yLimit: int) -> bool: 
        """
        Check whether this drop has passed the border given
        
        Return `True` if the object's y coordinate is 
        smaller than the given `yLimit`
        """
        #yLimit is 450
        #self.curPos[1] is -450 and increases

        return self.curPos[1] > -70
    
    #@Author AMELIA changed to make raindrops move downwards
    def set_next_position(self, speed: tuple[int, int]):
        """
        Set the next position of the object based on the speed
        """
        self.nextPos[0] = self.curPos[0] # + speed[0]
        self.nextPos[1] = self.curPos[1] + speed[1] #add speed instead of subtract

        self.curPos = self.nextPos 

    def get_position(self):
        """
        Return 4 points (in order: `y_offset`, `y_end`, `x_offset`, `x_end`) 
        determining the location of the current object (for this, we assume 
        objects are rectangular in shape)

        1. `x_offset`, `y_offset`: top left point of the image 
        2. `x_end`, `y_end`: bottom right point of the image 

        """
        x_offset = self.curPos[0]
        y_offset = self.curPos[1]
        x_end = x_offset + self.image.shape[1]
        y_end = y_offset + self.image.shape[0]
        return y_offset, y_end, x_offset, x_end 
