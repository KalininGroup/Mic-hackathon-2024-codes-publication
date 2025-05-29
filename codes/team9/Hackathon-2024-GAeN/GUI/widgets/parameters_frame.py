import tkinter as tk
import customtkinter as ctk

class Parameters_frame(ctk.CTkFrame):
    def __init__(self, root, parameters_list, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.grid_columnconfigure(1,weight=1)

        self.labels_list = []
        self.entrys_list = []
        self.variables_dic = {}

        for parameter in parameters_list:
            self.variables_dic[parameter] = tk.DoubleVar(self, 0)
            self.labels_list.append(ctk.CTkLabel(self, text = parameter, font = ('American typewriter', 18)))
            self.entrys_list.append(ctk.CTkEntry(self, textvariable=self.variables_dic[parameter], justify = 'right', font = ('American typewriter', 18)))
  
        i = 0
        for label, entry in zip(self.labels_list, self.entrys_list):
            label.grid(row = i, column = 0,  sticky = 'e', padx = 5, pady = 10) 
            entry.grid(row = i, column = 1,  padx = 5, pady = 10)
            i += 1