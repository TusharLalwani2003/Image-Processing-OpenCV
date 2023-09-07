# -------------------- Making Salt and Pepper Noise --------------------#

# Import Library
import cv2
import numpy as np
import random

# Defining Global Variables
ImagePath = "./test.png"        # Image Path
noiseFactor = 0.25              # How Much Noise should be generated (From 0 to 1)

# Defining Salting and Pepper Noise making function
def SaltAndPepperSprinkler(img, noiseFactor):
    # Making a New empty Image
    new_img = np.zeros(img.shape, np.uint8)

    # Go thorugh each pixel of the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Generate a Random Number that will decide whether noise is 
            # present on that pixel or not
            nf = random.random()

            # Either make the pixel white or black or no noise based on the random number
            if nf < (noiseFactor / 2):
                new_img[i, j] = [0, 0, 0]
            elif nf > (1 - (noiseFactor / 2)):
                new_img[i, j] = [255, 255, 255]
            else:
                new_img[i, j] = img[i, j]
    return new_img


# Read the image
img = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_COLOR_4)
img2 = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_GRAYSCALE_4)

# Convert to Noisy Image
img3 = SaltAndPepperSprinkler(img, noiseFactor)
img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Show all the images
cv2.imshow("OG Image : ", img)
cv2.imshow("GrayScale : ", img2)
cv2.imshow("Noisy Image : ", img3)
cv2.imshow("Grayscale Noisy Image : ", img4)

# Wait for the images to load and display
# Close when user closes the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
