import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# 1. Histogramme grayscale
# 2. Histogramme RGB
# 3. Equalization
# 4. CLAHE


"""
img = cv.imread('../imdata/car_4.jpg',cv.IMREAD_GRAYSCALE)
#plt.imshow(img,cmap='gray')
hist = cv.calcHist([img],[0],None,[256],[0,256])
#plt.plot(hist,color='blue')
#plt.show()
"""

img = cv.imread('../imdata/office_1.jpg',cv.IMREAD_GRAYSCALE)
#plt.imshow(img,cmap='gray')
hist = cv.calcHist([img],[0],None,[256],[0,256])  

img2 = cv.equalizeHist(img) #  Egalisation d'histogramme globale

# Egalisation d'histogramme adaptée CLAHE

clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
img3 = clahe.apply(img)

fig,ax = plt.subplots(1,3,figsize=(10,4))

# image
ax[0].imshow(img,cmap = 'gray')
ax[0].set_title('mon image originale')
ax[0].axis('off')


ax[1].imshow(img2,cmap='gray')
ax[1].set_title('mon image corrigee avec eglisation histogramme globale')

ax[2].imshow(img3,cmap='gray')
ax[2].set_title('mon image corrigee avec egalisation histogramme adaté')

plt.show()








# image
"""ax[0].imshow(img2_rgb)
ax[0].set_title('mon image originale')
ax[0].axis('off')
"""
"""color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img2],[i],None,[256],[0,256])
    ax[1].plot(histr,color = col)
    ax[1].set_title('Mon histogramme par canal')
    ax[1].set_xlabel('les intensités de 0 - 255')
    ax[1].set_ylabel('nombre de pixels')

plt.show()
"""