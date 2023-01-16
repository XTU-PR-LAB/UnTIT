import cv2
import numpy as np

y = [13,27,26,41,54,27,44,56,59,86,108,61,87,106,10,11,13,13]
x = [35,34,22,15,1,45,50,44,20,20,19,36,36,38,32,37,28,41]


if __name__ == '__main__':
    img = np.zeros((1024, 512, 3))
    for i in range(0, 18):
        cv2.circle(img, (x[i] * 8, y[i] * 8), 5, (255, 255, 255), 2)
        cv2.putText(img, str(i), (x[i] * 8 + 10, y[i] * 8 + 10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 255, 255), thickness = 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)