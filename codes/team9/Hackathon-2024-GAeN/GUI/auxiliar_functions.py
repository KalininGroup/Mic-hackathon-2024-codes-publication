import numpy as np
from PIL import Image, ImageTk

#Normalize a image/matrix
def normalize_image(image):
    minim = np.amin(image)
    maxim = np.amax(image)
    
    if(minim != maxim):
       normalize_matrix = (image - minim) / (maxim - minim)
    elif(maxim != 0):
       normalize_matrix = image/maxim
    else:
       normalize_matrix = image
   
    return normalize_matrix

# Convert a matrix/image in a picture     
def matrix_to_picture(ventana, matrix, size):
    matrix = normalize_image(matrix)
    image = (255*matrix).astype(np.uint8)
    image = Image.fromarray(image).resize(size)
    picture = ImageTk.PhotoImage(image, master = ventana)
    
    return picture


# Coordinates changing functions
def matrix_to_image_coordinates(matrix_coords, matrix_width, matrix_height, image_width, image_height):
    """
    Converts matrix coordinates to image coordinates.

    Args:
        matrix_coords (tuple): Matrix coordinates in the format (row, column).
        matrix_width (int): Width of the matrix.
        matrix_height (int): Height of the matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Image coordinates in the format (x, y).
    """
    row, column = matrix_coords

    # Calculate the scaling factors to map matrix coordinates to image coordinates
    x_scale = image_width / matrix_width
    y_scale = image_height / matrix_height

    # Calculate the corresponding image coordinates
    x = column * x_scale
    y = row * y_scale

    # Make sure the coordinates are within the image bounds
    x = max(0, min(x, image_height - 1))
    y = max(0, min(y, image_width - 1))

    return x, y

def image_to_matrix_coordinates(image_coords, image_width, image_height, matrix_width, matrix_height):
    """
    Converts image coordinates to matrix coordinates.

    Args:
        image_coords (tuple): Image coordinates in the format (x, y).
        matrix_width (int): Width of the matrix.
        matrix_height (int): Height of the matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Matrix coordinates in the format (row, column).
    """
    x, y = image_coords

    # Calculate the scaling factors to map image coordinates to matrix coordinates
    x_scale = matrix_width / image_width
    y_scale = matrix_height / image_height

    # Calculate the corresponding matrix coordinates
    column = int(x * x_scale + 0.5) # round to integer
    row = int(y * y_scale + 0.5) # round to integer

    # Make sure the coordinates are within the matrix bounds
    column = max(0, min(column, matrix_height - 1))
    row = max(0, min(row, matrix_width - 1))

    return row, column


#Save a matrix as .png (image)
def save_matrix_as_image(matrix, name, quality = 16, _format = '.png'):
    
    matrix = normalize_image(matrix)
    matrix = matrix*(2**quality-1)
    
    if quality == 16:
        matrix = np.uint16(matrix)
    if quality == 8:
        matrix = np.uint8(matrix)
        
    image = Image.fromarray(matrix)
    image.save(name + _format)
