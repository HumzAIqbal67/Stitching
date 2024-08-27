import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread('captured_imagesNS-DLU\image_000.jpg')
image2 = cv2.imread('captured_imagesNS-DLU\image_001.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the keypoints on both images
image1_keypoints = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_keypoints = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Plot keypoints
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Image 1")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image2_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Keypoints in Image 2")
plt.show()

# Draw the matches
matches_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot matches
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
plt.title("Matches between Images")
plt.show()

# Extract the matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# Estimate the translation between the images using the matches
translation_matrix, _ = cv2.estimateAffinePartial2D(points1, points2)

# Get the dimensions of the images
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

# Create a canvas large enough to fit both images
canvas_height = max(height1, height2)
canvas_width = width1 + width2
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Place the first image on the canvas
canvas[:height1, :width1] = image1

# Apply the translation to the second image and place it on the canvas
translated_image2 = cv2.warpAffine(image2, translation_matrix, (canvas_width, canvas_height))
canvas[:height2, width1:width1 + width2] = translated_image2[:height2, :width2]

# Plot the final stitched image
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("Stitched Image")
plt.show()

# Plot the vectors
plt.figure(figsize=(10, 10))
for i in range(len(points1)):
    plt.plot([points1[i][0], points2[i][0] + width1], [points1[i][1], points2[i][1]], 'r-', lw=1)
plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
plt.title("Feature Matching Vectors")
plt.show()
