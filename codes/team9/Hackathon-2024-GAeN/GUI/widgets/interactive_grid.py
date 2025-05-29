import numpy as np
from widgets.interactive_image import Interactive_image
from auxiliar_functions import image_to_matrix_coordinates, matrix_to_image_coordinates

class Interactive_grid(Interactive_image):
    def __init__(self, root, command, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.command = command
        self.square_size = 256 * self.height/self.matrix.shape[0]
        self.rectangle = self.canvas.create_rectangle((0, 0), (0, 0), outline='yellow', width= 3, tags='rectangle')
        self.create_selector(self.height/2, self.width/2)
        

        self.canvas.bind('<1>', lambda e: self.create_selector(e.x, e.y))
        self.canvas.bind('<B1-Motion>', lambda e: self.create_selector(e.x, e.y))


    def get_coord(self, e):
        print(e)

    def creat_selector_matrix(self, x_m, y_m):
        x, y = matrix_to_image_coordinates((x_m, y_m),  *self.matrix.shape, self.height, self.width)
        self.create_selector(x, y)
    
    def create_selector(self, x, y, square = False, outline = 'yellow'):
        self.canvas.delete(self.rectangle)

        x0, xf = x - self.square_size/2, x + self.square_size/2
        y0, yf = y - self.square_size/2, y + self.square_size/2
    
        self.rectangle = self.canvas.create_rectangle((x0, y0), (xf,yf),  outline=outline, width= 1, tags='rectangle')

        x_m, y_m = image_to_matrix_coordinates((x,y), self.height, self.width, *self.matrix.shape)
        matrix_crop = self.matrix[x_m - 128 : x_m + 128, y_m - 128 : y_m + 128]
        # print(x,y,x_m,y_m)

        self.command(matrix_crop)

       