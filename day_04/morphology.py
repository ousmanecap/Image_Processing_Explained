import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Erosion, # Dilatation (les deux operations de base)
# Les opérations morphologiques avancées (ouverture, fermeture, gradient morphologique)

BASE_DIR = os.getcwd()
img_path = os.path.join(BASE_DIR, 'imdata', 'hands2.jpg')

img = cv.imread(img_path,cv.IMREAD_GRAYSCALE) # obligatoire pour le seuillage

# pourquoi le seuillage parce qu'on veut avoir une image binaire
img_th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2) # me donne une image binaire

# appliquer erosion , dilatation
size = 3
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(size,size)) # element structure en fonction de la forme et du noyau
img_ero = cv.erode(img_th,kernel,iterations = 1) # pour erosion

img_dila = cv.dilate(img_th,kernel,iterations = 1) #pour la dilatation

#figure

"""plt.figure(figsize=(10,4))
plt.subplot(131)
plt.imshow(img_th,cmap='gray')
plt.title('Orginal seuillee')
plt.axis('off')

plt.subplot(132)
plt.imshow(img_ero,cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_dila,cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.show()
"""
# Morphologie avancée (ouverture, fermeture, gradient morphologique)
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(size,size))
img_open = cv.morphologyEx(img_th,cv.MORPH_OPEN,kernel1) # ouverture = supprimer les petits objets

img_fer = cv.morphologyEx(img_th,cv.MORPH_CLOSE,kernel1) # Fermeture = combler les trous

img_gra = cv.morphologyEx(img_th,cv.MORPH_GRADIENT,kernel1) # Gradient morphologique pour le contour

plt.figure(figsize=(10,4))
plt.subplot(141)
plt.imshow(img_open,cmap='gray')
plt.title('Ouverture')
plt.axis('off')

plt.subplot(142)
plt.imshow(img_fer,cmap='gray')
plt.title('Fermeture')
plt.axis('off')

plt.subplot(143)
plt.imshow(img_gra,cmap='gray')
plt.title('Gradient(contour)')
plt.axis('off')

plt.subplot(144)
plt.imshow(img_th,cmap='gray')
plt.title('image binaire')
plt.axis('off')

plt.show()