from typing import Tuple
import customtkinter as ctk

class Frame_titled(ctk.CTkFrame):
    def __init__(self, root, title, font = ('American typewriter', 24), *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        self.title_label = ctk.CTkLabel(self, text=title, font = font)
        self.title_label.pack(padx = 5, pady = 25)