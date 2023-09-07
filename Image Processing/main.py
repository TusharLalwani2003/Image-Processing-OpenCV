from cv2 import threshold
import IPH
import cv2
import numpy as np
import matplotlib.pyplot as plt


ImagePath = "test.png"
img = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_COLOR_4)
img_gray = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_GRAYSCALE_4)

# # Split Image into RGB seprately
# B, G, R = cv2.split(img)
# IPH.DisplayMultiple({"Original": img,
#                      "Red": R,
#                      "Green": G,
#                      "Blue": B})

# # Resizing Image
# half = IPH.Scale(img, 0.5, 0.5)
# bigger = IPH.Scale(img, 2, 2)
# strech = IPH.Resize(img, 200, 400, cv2.INTER_NEAREST)
# IPH.DisplayMultiple({"Original": img,
#                      "Half": half,
#                      "Bigger": bigger,
#                      "Strech": strech})

# # Edge Detection
# edge = IPH.Outline(img)
# IPH.DisplayMultiple({"Original": img,
#                      "Border": edge})

# # Blurring
# Gaussian = IPH.Blur(img, 75, 75)
# Median = IPH.Blur(img, 5)
# Bilateral = IPH.Blur(img, 9, 75, 75)
# IPH.DisplayMultiple({"Original": img,
#                      "Gaussian": Gaussian,
#                      "Median": Median,
#                      "Bilateral": Bilateral})


# # Erosion
# Eroded = IPH.Erode(img, 5, 5, 1)
# IPH.DisplayMultiple({"Original": img,
#                      "Eroded": Eroded})

# # Dilation
# Dilated = IPH.Dilate(img, 5, 5, 1)
# IPH.DisplayMultiple({"Original": img,
#                      "Dilated": Dilated})

# # Grayscale Intensity Hystogram
# IPH.HistDisplay(img)

# # Threshold
# IPH.DisplayMultiple({'Original': img,
#                      'Binary Threshold': IPH.BinaryThresh(img),
#                      'Invert Binary Threshold': IPH.InvBinaryThresh(img),
#                      'Trunc Threshold': IPH.TruncThresh(img),
#                      'To Zero Threshold': IPH.ToZeroThresh(img),
#                      'Invert To Zero Threshold': IPH.InvToZeroThresh(img)})

# # Adaptive threshold
# IPH.DisplayMultiple({"Original": img,
#                      "Grayscale": IPH.Grayscale(img),
#                      "Adaptive": IPH.AdaptiveThresh(img)})

# # Otsu Threshold
# IPH.DisplayMultiple({"Original": img,
#                      "Grayscale": IPH.Grayscale(img),
#                      "Otsu Threshold": IPH.OtsuThresh(img)})

# # All Threshold
# IPH.DisplayMultiple({'Original': img,
#                      'Binary Threshold': IPH.BinaryThresh(img),
#                      'Invert Binary Threshold': IPH.InvBinaryThresh(img),
#                      'Trunc Threshold': IPH.TruncThresh(img),
#                      'To Zero Threshold': IPH.ToZeroThresh(img),
#                      'Invert To Zero Threshold': IPH.InvToZeroThresh(img),
#                      'Adaptive': IPH.AdaptiveThresh(img),
#                      'Otsu Threshold': IPH.OtsuThresh(img)})

# # Color Filter
# lower = np.array([0, 150, 100])
# upper = np.array([10, 255, 255])
# IPH.DisplayMultiple({"Original": img,
#                      "Filtered": IPH.colorFilter(img, lower, upper)})

# # Hough Lines
# res, hough = IPH.HoughLines(img)
# if res == "Found":
#     IPH.DisplayMultiple({"Original": img,
#                         "Hough": hough})
# else:
#     print("No Lines")


cv2.destroyAllWindows()
