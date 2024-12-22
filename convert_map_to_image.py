import numpy as np
from PIL import Image
import os

def load_landscape(file_path, row_length):
    """Load the landscape from a file and convert it into a 2D numpy array."""
    with open(file_path, 'r') as f:
        data = f.read().strip()
    
    # Split into rows and clean each row
    rows = []
    for i in range(0, len(data), row_length):
        row = data[i:i+row_length].strip()
        # Extract only digits and pad with zeros if necessary
        digits = [int(char) for char in row if char.isdigit()]
        # Ensure each row has exactly row_length elements
        if len(digits) < row_length:
            digits.extend([0] * (row_length - len(digits)))
        elif len(digits) > row_length:
            digits = digits[:row_length]
        rows.append(digits)
    
    return np.array(rows)

def render_landscape(landscape, tile_images):
    """Render the landscape as an image using tile images."""
    tile_height, tile_width = tile_images[0].size
    rows, cols = landscape.shape
    canvas = Image.new("RGBA", (cols * tile_width, rows * tile_height))
    
    for r in range(rows):
        for c in range(cols):
            tile_value = landscape[r, c]
            tile_image = tile_images.get(tile_value, tile_images[0])  # Default to empty tile
            canvas.paste(tile_image, (c * tile_width, r * tile_height))
    
    return canvas

def check_tile_files(tile_mapping):
    """Verify all tile image files exist."""
    missing_files = []
    for tile_type, path in tile_mapping.items():
        if not os.path.exists(path):
            missing_files.append(path)
    if missing_files:
        raise FileNotFoundError(f"Missing tile images: {missing_files}")

def main():
    # Correct tile mapping based on TILE_TYPES meanings
    tile_mapping = {
        0: "data/empty.png",    # Empty
        1: "data/river.png",    # Water (changed from river.png)
        2: "data/grass.png",    # Grass
        3: "data/rock.png",     # Rocks
        4: "data/mountain.png"  # Mountains (changed from riverstone.png)
    }
    
    # Verify tile images exist
    check_tile_files(tile_mapping)
    
    try:
        tile_images = {key: Image.open(path) for key, path in tile_mapping.items()}
        
        # Use correct row length (MAP_WIDTH from generate_map.py)
        row_length = 50  # MAP_WIDTH
        for i in range(10):
            # Load the landscape
            landscape = load_landscape(f"generated_map/generated_landscape_{i}.txt", row_length)
            
            # Render the landscape
            rendered_image = render_landscape(landscape, tile_images)
            
            # Save the output
            rendered_image.save(f"output/generated_landscape_image_{i}.png")
            print(f"Landscape image saved as 'generated_landscape_image_{i}.png'.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
