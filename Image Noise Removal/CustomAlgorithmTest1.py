# Importing Libraries
import cv2
import numpy as np

# Defining Global Variables
# ImagePath = "./test.png"
ImagePath = "./RealLifeTest.jpg"
# ImagePath = "./RealLifeTest2.jpg"
# ImagePath = "./RealLifeTest3.jpg"

# Defining Preference Matrix
def PreferenceMatrixMaker(img, lightness):
	prefMatrix = np.zeros([img.shape[0], img.shape[1]], np.uint8)
	results = np.zeros(9)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			prefV = 0
			if i != 0:
				deltaB = int(abs(int(img[i, j, 0]) - int(img[i - 1, j, 0])))
				deltaG = int(abs(int(img[i, j, 1]) - int(img[i - 1, j, 1])))
				deltaR = int(abs(int(img[i, j, 2]) - int(img[i - 1, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if j != 0:
				deltaB = int(abs(int(img[i, j, 0]) - int(img[i, j - 1, 0])))
				deltaG = int(abs(int(img[i, j, 1]) - int(img[i, j - 1, 1])))
				deltaR = int(abs(int(img[i, j, 2]) - int(img[i, j - 1, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if i != (img.shape[0] - 1):
				deltaB = int(abs(int(img[i + 1, j, 0]) - int(img[i, j, 0])))
				deltaG = int(abs(int(img[i + 1, j, 1]) - int(img[i, j, 1])))
				deltaR = int(abs(int(img[i + 1, j, 2]) - int(img[i, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if j != (img.shape[1] - 1):
				deltaB = int(abs(int(img[i, j + 1, 0]) - int(img[i, j, 0])))
				deltaG = int(abs(int(img[i, j + 1, 1]) - int(img[i, j, 1])))
				deltaR = int(abs(int(img[i, j + 1, 2]) - int(img[i, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if i != 0 and j != 0:
				deltaB = int(abs(int(img[i - 1, j - 1, 0]) - int(img[i, j, 0])))
				deltaG = int(abs(int(img[i - 1, j - 1, 1]) - int(img[i, j, 1])))
				deltaR = int(abs(int(img[i - 1, j - 1, 2]) - int(img[i, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if i != 0 and j != img.shape[1] - 1:
				deltaB = int(abs(int(img[i - 1, j + 1, 0]) - int(img[i, j, 0])))
				deltaG = int(abs(int(img[i - 1, j + 1, 1]) - int(img[i, j, 1])))
				deltaR = int(abs(int(img[i - 1, j + 1, 2]) - int(img[i, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if i != img.shape[0] - 1  and j != 0:
				deltaB = int(abs(int(img[i + 1, j - 1, 0]) - int(img[i, j, 0])))
				deltaG = int(abs(int(img[i + 1, j - 1, 1]) - int(img[i, j, 1])))
				deltaR = int(abs(int(img[i + 1, j - 1, 2]) - int(img[i, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			if i != img.shape[0] - 1 and j != img.shape[1] - 1:
				deltaB = int(abs(int(img[i + 1, j + 1, 0]) - int(img[i, j, 0])))
				deltaG = int(abs(int(img[i + 1, j + 1, 1]) - int(img[i, j, 1])))
				deltaR = int(abs(int(img[i + 1, j + 1, 2]) - int(img[i, j, 2])))
				if(deltaB + deltaG + deltaR <= lightness):
					prefV = prefV + 1
			prefMatrix[i, j] = prefV
			results[prefV] += 1
	# print(results)
	return prefMatrix

# Denoising the Image
def Denoiser(img, lightness, ExchangeThreshold, ConsiderationThreshold):
	prefMatrix = PreferenceMatrixMaker(img, lightness)
	new_img = np.zeros(img.shape, np.uint8)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			tempB = 0
			tempG = 0
			tempR = 0
			counter = 0
			if prefMatrix[i, j] <= ExchangeThreshold : 
				if i != 0 and prefMatrix[i - 1, j] >= ConsiderationThreshold:
					tempB += img[i - 1, j, 0]
					tempG += img[i - 1, j, 1]
					tempR += img[i - 1, j, 2]
					counter = counter + 1
				if j != 0 and prefMatrix[i, j - 1] >= ConsiderationThreshold:
					tempB += img[i, j - 1, 0]
					tempG += img[i, j - 1, 1]
					tempR += img[i, j - 1, 2]
					counter = counter + 1
				if i != img.shape[0] - 1 and prefMatrix[i + 1, j] >= ConsiderationThreshold:
					tempB += img[i + 1, j, 0]
					tempG += img[i + 1, j, 1]
					tempR += img[i + 1, j, 2]
					counter = counter + 1
				if j != img.shape[1] - 1 and prefMatrix[i, j + 1] >= ConsiderationThreshold:
					tempB += img[i, j + 1, 0]
					tempG += img[i, j + 1, 1]
					tempR += img[i, j + 1, 2]
					counter = counter + 1
				if i != 0 and j != 0 and prefMatrix[i - 1, j - 1] >= ConsiderationThreshold:
					tempB += img[i - 1, j - 1, 0]
					tempG += img[i - 1, j - 1, 1]
					tempR += img[i - 1, j - 1, 2]
					counter = counter + 1
				if i != 0 and j != img.shape[1] - 1 and prefMatrix[i - 1, j + 1] >= ConsiderationThreshold:
					tempB += img[i - 1, j + 1, 0]
					tempG += img[i - 1, j + 1, 1]
					tempR += img[i - 1, j + 1, 2]
					counter = counter + 1
				if i != img.shape[0] - 1 and j != 0 and prefMatrix[i + 1, j - 1] >= ConsiderationThreshold:
					tempB += img[i + 1, j - 1, 0]
					tempG += img[i + 1, j - 1, 1]
					tempR += img[i + 1, j - 1, 2]
					counter = counter + 1
				if i != img.shape[0] - 1 and j != img.shape[1] - 1 and prefMatrix[i + 1, j + 1] >= ConsiderationThreshold:
					tempB += img[i + 1, j + 1, 0]
					tempG += img[i + 1, j + 1, 1]
					tempR += img[i + 1, j + 1, 2]
					counter = counter + 1
				if counter == 0:
					new_img[i, j] = img[i, j]
				else:
					new_img[i, j] = [tempB / counter, tempG / counter, tempR / counter]
			else:
				new_img[i, j] = img[i, j]
	return new_img



# Reading Image
img = cv2.imread(ImagePath)
cv2.imshow("OG Image : ", img)

# # Denoising Image
# Lineancy = 100
# ExchangeThreshold = 2
# ConsiderationThreshold = 7
# for EXTH in range(3):
# 	for CSDRTH in range(3):
# 		ExchangeThreshold = 1 + EXTH
# 		ConsiderationThreshold = 6 + CSDRTH
# 		img2 = Denoiser(img, Lineancy, ExchangeThreshold, ConsiderationThreshold)
# 		cv2.imshow("First - " + str(ExchangeThreshold) + " " + str(ConsiderationThreshold), img2)
# 		for i in range(9):
# 			img2 = Denoiser(img2, Lineancy, ExchangeThreshold, ConsiderationThreshold)
# 		cv2.imshow("Last - " + str(ExchangeThreshold) + " " + str(ConsiderationThreshold), img2)

Lineancy = 100
ExchangeThreshold = 3
ConsiderationThreshold = 7
img2 = Denoiser(img, Lineancy, ExchangeThreshold, ConsiderationThreshold)
cv2.imshow("First - " + str(ExchangeThreshold) + " " + str(ConsiderationThreshold), img2)
for i in range(9):
	img2 = Denoiser(img2, Lineancy, ExchangeThreshold, ConsiderationThreshold)
cv2.imshow("Last - " + str(ExchangeThreshold) + " " + str(ConsiderationThreshold), img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
