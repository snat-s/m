# atkinson dithering because it really is what i like
# /// script
# dependencies = [
#   "opencv-python",
#   "numpy",
#   "tqdm",
# ]
# ///
import argparse
import cv2
import numpy as np
from tqdm.auto import trange

def load_image(img_path): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.float32) / 255.0 # normalize
    return img

def load_image_tiny(img_path, tiny_size=1024): 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate scaling to fit within tiny_size x tiny_size
    h, w = img.shape
    scale = min(tiny_size/w, tiny_size/h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    img = cv2.resize(img, (new_w, new_h))
    
    # Pad to 255x255 with black borders
    top = (tiny_size - new_h) // 2
    bottom = tiny_size - new_h - top
    left = (tiny_size - new_w) // 2
    right = tiny_size - new_w - left
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    img = np.array(img, dtype=np.float32) / 255.0
    return img

def save_image(img_path, img): 
    img = np.clip(img, 0, 1) * 255.0
    img = np.array(img, dtype=np.uint8)
    cv2.imwrite(img_path, img)

def atkinson(img):
    height, width = img.shape
    atkinson_dithered = img.copy()
    
    for i in trange(height):
        for j in range(width):
            old_pixel = atkinson_dithered[i][j]
            new_pixel = 1 if old_pixel >= 0.5 else 0
            atkinson_dithered[i][j] = new_pixel
            q_error = old_pixel - new_pixel
            
            if j + 1 < width and i + 1 < height and i + 1 < height and j + 1 < width and i + 1 < height and j - 1 >= 0 and i + 2 < height:
                atkinson_dithered[i, j+1] += q_error * 1/8
                atkinson_dithered[i+1, j] += q_error * 1/8
                atkinson_dithered[i+1, j+1] += q_error * 1/8
                atkinson_dithered[i+1, j-1] += q_error * 1/8
                atkinson_dithered[i+2, j] += q_error * 1/8
            
            # Up-right: (i-1, j+1) - but we skip this in raster scan order
            # as we've already processed that pixel
             
    return atkinson_dithered

def main():
    img_path = "./image.jpeg"
    output_path = "./ditheridu.jpg"
    img = load_image(img_path)
    atkinson_img = atkinson(img)
    save_image(output_path, atkinson_img)

if __name__ == "__main__":
    main()
