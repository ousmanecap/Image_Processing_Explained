
# Edge detection
# Sobel Operator
# Canny Edge detector


import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.getcwd()
img_path = os.path.join(BASE_DIR, 'imdata', 'cameraman.tif')

img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
assert img is not None, 'image non chargee'
"""plt.imshow(img,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir des valeurs
plt.show()
"""
# Flou gaussien
img_blur = cv.GaussianBlur(img,(5,5),0)

# Operateur de Sobel
"""sobelx = cv.Sobel(img_blur,cv.CV_64F,1,0,ksize=5) # axe vertical prédominant
sobely = cv.Sobel(img_blur,cv.CV_64F,0,1,ksize=5) # axe horizontal prédominant
sobelxy= cv.Sobel(img_blur,cv.CV_64F,1,1,ksize=5) #on s'interesse à tous les axes

plt.subplot(1,3,1),plt.imshow(sobelx,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir les valeurs
plt.title('axe vertical')

plt.subplot(1,3,2),plt.imshow(sobely,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir les valeurs
plt.title('axe horizontal')

plt.subplot(1,3,3),plt.imshow(sobelx,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir les valeurs
plt.title('sur tout')

plt.show()
"""
sobelxy= cv.Sobel(img_blur,cv.CV_64F,1,1,ksize=5) #on s'interesse à tous les axes
# Canny detector
img_canny = cv.Canny(img_blur,90,150) # val_min = 110 et val_max = 210
plt.subplot(1,3,1),plt.imshow(sobelxy,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir les valeurs
plt.title('sobel')

plt.subplot(1,3,2),plt.imshow(img_canny,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir les valeurs
plt.title('canny')

plt.subplot(1,3,3),plt.imshow(img,cmap='gray')
plt.xticks([]),plt.yticks([]) #pour ne pas avoir les valeurs
plt.title('img en gray scale')

plt.show()