import tkinter as tk
import customtkinter as ctk
from PIL import Image

from auxiliar_functions import matrix_to_picture, image_to_matrix_coordinates

class Interactive_image(ctk.CTkFrame):
    def __init__(self, root, matrix, width=500, height=500, title = '', font=('American typewriter', 20), *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.root = root
        self.width = width
        self.height = height
        self.matrix = matrix.astype('float32')

        # Zoom settings
        self.zoom_value = 1
        self.zoom_width = 100
        self.zoom_height = 100
        self.zoom_activated = False
        
        # Title
        self.title = ctk.CTkLabel(self, text=title, fg_color="transparent", font=font)
        if title != '':
            self.title.pack()

        # Image and Canvas    
        self.canvas = tk.Canvas(self, width=width, height=height, background='#2b2b2b')
        self.canvas_image = self.canvas.create_image((0, 0), anchor= 'nw')
        self.change_image(self.matrix)
        
        # Layout
        self.configure(fg_color="transparent")
        self.canvas.pack(padx = 10, pady = 5)

        # Interactions
        self.canvas.bind('<3>', lambda e: self.menu.post(e.x_root, e.y_root))
        self.canvas.bind('<MouseWheel>', self.wheel_zoom)

    def change_image(self, matrix):
        self.canvas.delete('spot')
        self.canvas.delete('ROI')

        self.matrix = matrix
        self.image = Image.fromarray(self.matrix).resize((self.width, self.height))
        self.picture = matrix_to_picture(self.root, self.matrix, size= (self.width, self.height))
        self.canvas.itemconfigure(self.canvas_image , image = self.picture)      

    def wheel_zoom(self, event):
        if self.zoom_value == 1:
            self.create_zoom(event)
            self.canvas.bind("<Motion>", self.zoom)
            self.canvas.configure(cursor="none")

        self.zoom_value += event.delta/240

        if self.zoom_value > 1:
            self.zoom(event)

        else:
           self.zoom_value = 1
           self.destroy_zoom(event) 
           self.canvas.unbind("<Motion>")
           self.canvas.configure(cursor="")
    
    def zoom(self,event):
        # Get the coordinates of the mouse
        x = event.x
        y = event.y

        # We calculate square coordinates
        x1 = max(0, x - self.zoom_width // self.zoom_value// 2)
        y1 = max(0, y - self.zoom_height // self.zoom_value // 2)
        x2 = min(self.image.width, x + self.zoom_width // self.zoom_value // 2)
        y2 = min(self.image.height, y + self.zoom_height // self.zoom_value // 2)
        
        # Change to matrix coordinates
        x0, y0 = image_to_matrix_coordinates((x1, y1),self.width, self.height, self.matrix.shape[1], self.matrix.shape[0])
        xf, yf = image_to_matrix_coordinates((x2, y2),self.width, self.height, self.matrix.shape[1], self.matrix.shape[0])

        # Show the zoom image in the zoom square
        self.canvas.image = matrix_to_picture(self.root, self.matrix[x0:xf, y0:yf], (self.zoom_width , self.zoom_height ))
        self.canvas.itemconfigure(self.zoom_item, image=self.canvas.image, anchor=tk.CENTER)
        self.canvas.coords(self.zoom_item, x, y)

    def destroy_zoom(self,event):
        self.canvas.delete(self.zoom_item)
    
    def create_zoom(self, event):
        self.zoom_item = self.canvas.create_image(event.x, event.y, anchor=tk.CENTER)
        self.zoom(event)

    