import numpy as np

def build_spiral_big(num_cells_x, num_cells_y, cell_size=4096):
        
        start_x = round(num_cells_x / 2) - 1
        start_y = num_cells_y // 2 
    
        coord_desordenado = []
        lista_id = []
        coord_ordenado = []

        grid_array = np.zeros((num_cells_y, num_cells_x), dtype=int)
        
        directions = [(1, 0), (0, -1), (-1, 0), (0, 1)] 
        direction_index = 0

        x, y = start_x, start_y
        cell_id = 1
        step_size = 1 
        steps_taken = 0
        direction_changes = 0  
        num_total = num_cells_x * num_cells_y

        while cell_id <= num_total:
            grid_array[y, x] = cell_id
            cell_id += 1
            
            dx, dy = directions[direction_index]
            x += dx
            y += dy
            steps_taken += 1

            if steps_taken == step_size:
                steps_taken = 0
                direction_index = (direction_index + 1) % 4
                direction_changes += 1

                if direction_changes % 2 == 0:
                    step_size += 1

        image_height = num_cells_y * cell_size
        image_width = num_cells_x * cell_size
        image = np.zeros((image_height, image_width), dtype=int)

        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell_value = grid_array[i, j]
                lista_id.append(cell_value)
                y_start, y_end = i * cell_size, (i + 1) * cell_size
                x_start, x_end = j * cell_size, (j + 1) * cell_size
                image[y_start:y_end, x_start:x_end] = cell_value
                coord_desordenado.append((y_start, x_start))

        coord_ordenado = [coord_desordenado[id_ - 1] for id_ in lista_id]
        coord_ordenado = [coord_desordenado[lista_id.index(i)] for i in range(1, len(coord_desordenado) + 1)]

        return image, coord_ordenado