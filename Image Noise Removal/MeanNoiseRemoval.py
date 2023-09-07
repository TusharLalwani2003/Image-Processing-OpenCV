# Importing Libraries
import cv2
import numpy as np
import random

# Defining global variables
ImagePath = "./test.png"
noiseFactor = 0.99

# Defining function to cause noise
def SaltAndPepperSprinkler(img, noiseFactor):
    noiseFactor = noiseFactor / 2
    new_img = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            nf = random.random()
            if nf < noiseFactor:
                new_img[i, j] = [0, 0, 0]
            elif nf > (1 - noiseFactor):
                new_img[i, j] = [255, 255, 255]
            else:
                new_img[i, j] = img[i, j]
    return new_img

# Defining function to remove noise from gray scale images
def GrayMedianNoiseRemove(gray_img):
    new_img = np.zeros(gray_img.shape, np.uint8)
    X = len(img[0])
    Y = len(img)
    new_element = 0
    for i in range(Y):
        for j in range(X):
            new_element = 0
            sizing = 0
            if gray_img[i, j] == 0 or gray_img[i, j] == 255:
                if i != 0 and gray_img[i - 1, j] != 0 and gray_img[i - 1, j] != 255:
                    new_element += gray_img[i - 1, j]
                    sizing = sizing + 1
                if j != 0 and gray_img[i, j - 1] != 0 and gray_img[i, j - 1] != 255:
                    new_element += gray_img[i, j - 1]
                    sizing = sizing + 1
                if i != 0 and j != 0 and gray_img[i - 1, j - 1] != 0 and gray_img[i - 1, j - 1] != 255:
                    new_element += gray_img[i - 1, j - 1]
                    sizing = sizing + 1
                if i != Y - 1 and gray_img[i + 1, j] != 0 and gray_img[i + 1, j] != 255:
                    new_element += gray_img[i + 1, j]
                    sizing = sizing + 1
                if j != X - 1 and gray_img[i, j + 1] != 0 and gray_img[i, j + 1] != 255:
                    new_element += gray_img[i, j + 1]
                    sizing = sizing + 1
                if i != Y - 1 and j != X - 1 and gray_img[i + 1, j + 1] != 0 and gray_img[i + 1, j + 1] != 255:
                    new_element += gray_img[i + 1, j + 1]
                    sizing = sizing + 1
                if i != 0 and j != X - 1 and gray_img[i - 1, j + 1] != 0 and gray_img[i - 1, j + 1] != 255:
                    new_element += gray_img[i - 1, j + 1]
                    sizing = sizing + 1
                if i != Y - 1 and j != 0 and gray_img[i + 1, j - 1] != 0 and gray_img[i + 1, j - 1] != 255:
                    new_element += gray_img[i + 1, j - 1]
                    sizing = sizing + 1
                if sizing != 0:
                    new_img[i, j] = new_element/sizing
            else:
                new_img[i, j] = gray_img[i, j]
    return new_img


# Defining function to remove noise from colored images
def ColorMedianNoiseRemove(color_img):
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    new_img = np.zeros(color_img.shape, np.uint8)
    X = len(img[0])
    Y = len(img)
    new_element_B = 0
    new_element_G = 0
    new_element_R = 0
    for i in range(Y):
        for j in range(X):
            new_element_B = 0
            new_element_G = 0
            new_element_R = 0
            sizing = 0
            if gray_img[i, j] == 0 or gray_img[i, j] == 255:
                if i != 0 and gray_img[i - 1, j] != 0 and gray_img[i - 1, j] != 255:
                    new_element_B += color_img[i - 1, j, 0]
                    new_element_G += color_img[i - 1, j, 1]
                    new_element_R += color_img[i - 1, j, 2]
                    sizing = sizing + 1
                if j != 0 and gray_img[i, j - 1] != 0 and gray_img[i, j - 1] != 255:
                    new_element_B += color_img[i, j - 1, 0]
                    new_element_G += color_img[i, j - 1, 1]
                    new_element_R += color_img[i, j - 1, 2]
                    sizing = sizing + 1
                if i != 0 and j != 0 and gray_img[i - 1, j - 1] != 0 and gray_img[i - 1, j - 1] != 255:
                    new_element_B += color_img[i - 1, j - 1, 0]
                    new_element_G += color_img[i - 1, j - 1, 1]
                    new_element_R += color_img[i - 1, j - 1, 2]
                    sizing = sizing + 1
                if i != Y - 1 and gray_img[i + 1, j] != 0 and gray_img[i + 1, j] != 255:
                    new_element_B += color_img[i + 1, j, 0]
                    new_element_G += color_img[i + 1, j, 1]
                    new_element_R += color_img[i + 1, j, 2]
                    sizing = sizing + 1
                if j != X - 1 and gray_img[i, j + 1] != 0 and gray_img[i, j + 1] != 255:
                    new_element_B += color_img[i, j + 1, 0]
                    new_element_G += color_img[i, j + 1, 1]
                    new_element_R += color_img[i, j + 1, 2]
                    sizing = sizing + 1
                if i != Y - 1 and j != X - 1 and gray_img[i + 1, j + 1] != 0 and gray_img[i + 1, j + 1] != 255:
                    new_element_B += color_img[i + 1, j + 1, 0]
                    new_element_G += color_img[i + 1, j + 1, 1]
                    new_element_R += color_img[i + 1, j + 1, 2]
                    sizing = sizing + 1
                if i != 0 and j != X - 1 and gray_img[i - 1, j + 1] != 0 and gray_img[i - 1, j + 1] != 255:
                    new_element_B += color_img[i - 1, j + 1, 0]
                    new_element_G += color_img[i - 1, j + 1, 1]
                    new_element_R += color_img[i - 1, j + 1, 2]
                    sizing = sizing + 1
                if i != Y - 1 and j != 0 and gray_img[i + 1, j - 1] != 0 and gray_img[i + 1, j - 1] != 255:
                    new_element_B += color_img[i + 1, j - 1, 0]
                    new_element_G += color_img[i + 1, j - 1, 1]
                    new_element_R += color_img[i + 1, j - 1, 2]
                    sizing = sizing + 1
                if sizing != 0:
                    new_img[i, j] = [new_element_B/sizing,
                                     new_element_G/sizing, new_element_R/sizing]
                else:
                    new_img[i, j] = [0, 0, 0]
            else:
                new_img[i, j] = color_img[i, j]
    return new_img


img = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_COLOR_4)
img2 = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_GRAYSCALE_4)

img3 = SaltAndPepperSprinkler(img, noiseFactor)
img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img5 = GrayMedianNoiseRemove(img4)
img6 = GrayMedianNoiseRemove(img5)
for i in range(9):
    img6 = GrayMedianNoiseRemove(img6)
img7 = ColorMedianNoiseRemove(img3)
img8 = ColorMedianNoiseRemove(img7)
for i in range(9):
    img8 = ColorMedianNoiseRemove(img8)

cv2.imshow("OG Image : ", img)
cv2.imshow("GrayScale : ", img2)
cv2.imshow("Noisy Image : ", img3)
cv2.imshow("Gray Noisy Image : ", img4)
cv2.imshow("Cancelled Gray Noisy Image : ", img5)
cv2.imshow("Cancelled Gray Noisy Image Iteration 10 times: ", img6)
cv2.imshow("Cancelled Color Noisy Image : ", img7)
cv2.imshow("Cancelled Color Noisy Image Iteration 10 times: ", img8)


cv2.waitKey(0)
cv2.destroyAllWindows()
