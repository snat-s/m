import cv2
import numpy as np 

def main(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img[img >= 200] = 1
    img[img < 200] = 0
    height, width = img.shape
    x, y = np.mgrid[:height, :width]
    data = np.column_stack((x.ravel(), y.ravel(), img.ravel()))
    np.savetxt('train.csv', data, delimiter=',', fmt='%d')

if __name__ == '__main__':
    main('tiniest_logo.png')