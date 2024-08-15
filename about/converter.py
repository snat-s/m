import cv2
import numpy as np 

def main(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    x, y = np.mgrid[:height, :width]
    data = np.column_stack((x.ravel(), y.ravel(), img.ravel()))
    np.savetxt('train.csv', data, delimiter=',', fmt='%d')

def csv_to_image(csv_path, output_path):
    # Read the CSV file
    data = np.genfromtxt(csv_path, delimiter=',', dtype=int)
    
    # Extract x, y, and pixel values
    x, y, pixel_values = data[:, 0], data[:, 1], data[:, 2]
    
    # Get image dimensions
    height, width = x.max() + 1, y.max() + 1
    
    # Create an empty image
    img = np.zeros((height, width), dtype=np.uint8)
    
    img[x, y] = pixel_values
    cv2.imwrite(output_path, img)
    
    print(f"Image saved to {output_path}")

if __name__ == '__main__':
    main('tiniest_logo.png')
    csv_to_image('train.csv', 'output_image.png')