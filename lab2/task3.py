import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
output_dir = 'processed_images'
os.makedirs(output_dir, exist_ok=True)

# Load images
img1 = cv2.imread('original_images/image1.jpg')
img2 = cv2.imread('original_images/image2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match keypoints using BFMatcher and KNN
matcher = cv2.BFMatcher()
knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Sort good matches by distance and select top matches
good_matches = sorted(good_matches, key=lambda x: x.distance)
top_matches = good_matches[:50]

# Draw keypoint matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, top_matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
for m in top_matches:
    pt1 = (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1]))
    pt2 = (int(keypoints2[m.trainIdx].pt[0] + img1.shape[1]), int(keypoints2[m.trainIdx].pt[1]))
    cv2.line(img_matches, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)  # Green, thick lines

cv2.imwrite(os.path.join(output_dir, 'img_matches.jpg'), img_matches)

# Show matched keypoints
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('Top 50 Keypoint Matches')
plt.axis('off')
plt.show()

# Extract keypoints from top matches
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

# Find homography using RANSAC
matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp one image to align with the other
result = cv2.warpPerspective(img1, matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

# Save and show the stitched image
cv2.imwrite(os.path.join(output_dir, 'stitched_img.jpg'), result)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Stitched Image')
plt.axis('off')
plt.show()