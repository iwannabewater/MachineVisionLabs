import cv2
import matplotlib.pyplot as plt
import os
from skimage.util import random_noise

# Ensure the output directory exists
output_dir = 'processed_images'
os.makedirs(output_dir, exist_ok=True)

# Load images
img1 = cv2.imread('original_images/image1.jpg')
img2 = cv2.imread('original_images/image2.jpg')

# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=20)

# Processed versions of the images
# a) Scaled by 120%
scaled1 = cv2.resize(img1, None, fx=1.2, fy=1.2)
scaled2 = cv2.resize(img2, None, fx=1.2, fy=1.2)

# b) Rotated clockwise by 60 degrees
rows1, cols1 = img1.shape[:2]
rows2, cols2 = img2.shape[:2]
M1 = cv2.getRotationMatrix2D((cols1/2, rows1/2), -60, 1)
M2 = cv2.getRotationMatrix2D((cols2/2, rows2/2), -60, 1)
rotated1 = cv2.warpAffine(img1, M1, (cols1, rows1))
rotated2 = cv2.warpAffine(img2, M2, (cols2, rows2))

# c) Contaminated with salt and pepper noise (keeping the color)
def add_salt_and_pepper_noise(image, amount=0.05):
    noisy_image = image.copy()
    for i in range(3):  # Apply noise to each channel separately
        noisy_channel = random_noise(noisy_image[:, :, i], mode='s&p', amount=amount)
        noisy_image[:, :, i] = (noisy_channel * 255).astype('uint8')
    return noisy_image

noisy1 = add_salt_and_pepper_noise(img1)
noisy2 = add_salt_and_pepper_noise(img2)

# Compute and draw SIFT keypoints
def compute_and_draw_sift(image, output_path, title):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    img_kp = cv2.drawKeypoints(image, keypoints, None, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(img_kp, (int(x), int(y)), 8, (0, 0, 255), 2, cv2.LINE_AA)  # Red, large, thick circles
    cv2.imwrite(output_path, img_kp)
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display the keypoints for each processed version
compute_and_draw_sift(scaled1, os.path.join(output_dir, 'scaled1_keypoints.jpg'), 'Scaled Image 1 - 20 Keypoints')
compute_and_draw_sift(scaled2, os.path.join(output_dir, 'scaled2_keypoints.jpg'), 'Scaled Image 2 - 20 Keypoints')
compute_and_draw_sift(rotated1, os.path.join(output_dir, 'rotated1_keypoints.jpg'), 'Rotated Image 1 - 20 Keypoints')
compute_and_draw_sift(rotated2, os.path.join(output_dir, 'rotated2_keypoints.jpg'), 'Rotated Image 2 - 20 Keypoints')
compute_and_draw_sift(noisy1, os.path.join(output_dir, 'noisy1_keypoints.jpg'), 'Noisy Image 1 - 20 Keypoints')
compute_and_draw_sift(noisy2, os.path.join(output_dir, 'noisy2_keypoints.jpg'), 'Noisy Image 2 - 20 Keypoints')