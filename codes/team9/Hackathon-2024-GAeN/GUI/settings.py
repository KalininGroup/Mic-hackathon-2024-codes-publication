import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from widgets.titled_frame import Frame_titled
from widgets.parameters_frame import Parameters_frame

class Frame_settings(ctk.CTkFrame):
    def __init__(self, root, project_path, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        ctk.CTkLabel(self, text = 'Settings',anchor = 'center', font=('American typewriter', 32), fg_color='#1f6aa5').pack(fill = 'x', padx = 5, pady = 5)
 
        # Frame Acquisition
        self.frame_acquisition = Frame_titled(self, 'Acquistion settings')
        self.parameters = Parameters_frame(self.frame_acquisition, ['# NPs images'])
        self.button_play = ctk.CTkButton(self.frame_acquisition, text = 'Start acquisition', cursor =  'hand2', fg_color='green', font=('American typewriter', 20))
        self.label_model1 = ctk.CTkLabel(self, text=' Probability of NPs: ', font=('American typewriter', 20))

        # Layout
        self.frame_acquisition.pack(padx = 5, pady = 10, fill = 'x')
        self.parameters.pack(padx = 5, pady = 10, fill = 'x')
        self.button_play.pack(padx = 5, pady = 10, fill = 'x')
        self.label_model1.pack(padx = 5, pady = 10)

