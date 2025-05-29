import numpy as np
from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *
from auxiliar_functions import normalize_image
from Sample_generetor import build_spiral_big

class Autonomous_navigator():
    def __init__(self, image_sample, large_x = 4, large_y = 3, size_grid = 1024):
        self.image_sample = image_sample
        self.image_spiral, self.grid_coordinates = build_spiral_big(large_x, large_y, cell_size = size_grid)
        self.path_tracking = []

        self.step = 32
        self.fov = 256
        self.threshold = 1.99
        self.nps_finded = 0

        self.num_step_x = 0
        self.num_step_y = 0
        self.num_step=(size_grid-self.fov)/self.step + 1

        n = large_x*large_y
        self.id_not_visited_yet = [i for i in range(n)]
        self.id_visited = []

        self.learn_inf = load_learner('ML/models/nanos_detector.pkl')
        # self.model = torch.load('ML/models/model_tecnai.pt')

    def calculate_np_probability(self, image):
        img_format = np.uint8(image*255) #We need change the format
        pred = self.learn_inf.predict(img_format)
        probability = float(pred[2][1])
        
        return probability


    def border(self, i, j):
        """
        This function detects borders when scanning the image.
        border_x and border_y: coordinates along the x and y axis that we want to verify if they are exceeding the limits of the images.
        
        """
        
        border_left = j
        border_right = j + self.fov
        border_up = i - self.fov
        border_down = i 
        dimension_down, dimension_right = self.image_sample.shape
        
        # if border_left >= 0 or border_right < dimension_right:
        #     return True

        # if border_up >= 0 or border_down < dimension_down : 
        #     return True

        return False

    def standard_movement(self, i, j):

        self.num_step_x+=1
        i_new = i + self.step
        j_new = j

        if(self.num_step_x == self.num_step):
            self.num_step_x=0
            self.num_step_y+=1
            i_new -= self.num_step*self.step
            j_new = j + self.step
            
        if(self.num_step_y == self.num_step):
            self.num_step_x=0
            self.num_step_y=0
            i_new, j_new = self.find_new_grid(i, j)
  
        return i_new, j_new
    
    def find_new_grid(self, i, j):
        """
        This function finds regions that haven't been explored yet
        """
        self.id_visited = list(set(self.image_spiral[i - self.fov: i, j : j + self.fov].flatten()))
        # self.id_not_visited_yet.remove(list(set(self.id_visited) & set(self.id_not_visited_yet)))

        return self.grid_coordinates[self.id_not_visited_yet[0]] 
    
    def find_np(self, p_np, coord_microscope):
        """
        This algorithm determines the navigation path for detecting nanoparticles. The direction of movement is guided by the probability of encountering nanoparticles at the next 
        step. This probability is provided by our model, which compares the likelihood of finding nanoparticles in different directions and selects the direction with the highest 
        probability.
        """
        n = self.step
        i, j = coord_microscope

        #Up
        if not self.border(i - n, j): 
            scanning_m_u = self.image_sample[i-self.fov-n:i-n, j:j+self.fov]
            prob_mov_u = self.calculate_np_probability(scanning_m_u)
        else:
            prob_mov_u = 0
        
        #Down
        if not self.border(i+n, j): 
            scanning_m_d = self.image_sample[i-self.fov+n:i+n, j:j+self.fov]
            prob_mov_d = self.calculate_np_probability(scanning_m_d)
        else: 
            prob_mov_d = 0

        #Left
        if not self.border(i, j-n): 
            scanning_m_l = self.image_sample[i-self.fov:i, j-n:j+self.fov-n]
            prob_mov_l = self.calculate_np_probability(scanning_m_l)
        else:
            prob_mov_l = 0
        
        #Right
        if not self.border(i, j+n):     
            scannig_m_r = self.image_sample[i-self.fov:i, j+n:j+self.fov+n]
            prob_mov_r = self.calculate_np_probability(scannig_m_r)
        else:
            prob_mov_r = 0
        
        probs = [prob_mov_r, prob_mov_l, prob_mov_u, prob_mov_d]
        print(probs, p_np)
        max_prob = np.max(probs)

        #Acquiring the image of the nanoparticle when the surroundings haven't higher porbability of finding a nanoparticle than the current posistion
        if max_prob <= p_np: 
            i_new, j_new = i, j 
        
        #Displacement
        elif max_prob == prob_mov_r: #go to the right
            i_new, j_new = i, j+n
            print('right')
            
        elif max_prob == prob_mov_l: #go to the left
            i_new, j_new = i, j-n
            print('left')

        elif max_prob == prob_mov_u: #go up
            i_new, j_new = i-n, j
            print('up')
                
        elif max_prob == prob_mov_d: #go down
            i_new, j_new = i+n, j
            print('down')

        return i_new, j_new

    def acquire_images(self, i, j):
        matrix = self.image_sample[i - self.fov: i, j : j + self.fov]
        prediction = self.model.predict(matrix)
        print(len(prediction[1][0]))
        x, y, _  = prediction[1][0][-1]
        i_new = i - self.fov/2 + x
        j_new = j + y - self.fov/2
        self.nps_finded += 1

        return i_new, j_new


    def autonomous_navigation(self, coord_microscope):
        i, j = coord_microscope
        image_microscope = self.image_sample[i - self.fov: i, j : j + self.fov]

        # Calculate probability
        p_np = self.calculate_np_probability(image_microscope)
        print(p_np)

        if p_np < self.threshold:
            # j_new = j + self.step
            # i_new = i
            i_new, j_new = self.standard_movement(i,j)

        else:
            i_new, j_new = self.find_np(p_np, coord_microscope)

            if (i == i_new) & (j == j_new):
                i_new, j_new = self.acquire_images(i, j)
                self.find_new_grid(i_new, j_new)

        self.path_tracking.append((i_new, j_new))
        return i_new, j_new



def get_x(r):
    im = Image.open('imagenes_M/' + r['labels'] +'/'+ r['file name'])
    return np.array(np.uint8(normalize_image(im)*255))
def get_y(r): return r['labels']