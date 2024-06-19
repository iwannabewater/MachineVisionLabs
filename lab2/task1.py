import cv2
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

# Draw keypoints on the images with large, thick red circles
def draw_keypoints(image, keypoints, output_path, title):
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

draw_keypoints(img1, keypoints1, os.path.join(output_dir, 'img1_keypoints.jpg'), 'Image 1 - All Keypoints')
draw_keypoints(img2, keypoints2, os.path.join(output_dir, 'img2_keypoints.jpg'), 'Image 2 - All Keypoints')

# Reduce number of keypoints to the 20 most prominent ones
# Here I use the contrastThreshold to reduce the keypoints
sift_limited = cv2.SIFT_create(nfeatures=20)
keypoints1_limited, descriptors1_limited = sift_limited.detectAndCompute(gray1, None)
keypoints2_limited, descriptors2_limited = sift_limited.detectAndCompute(gray2, None)

# Draw the reduced keypoints
draw_keypoints(img1, keypoints1_limited, os.path.join(output_dir, 'img1_keypoints_limited.jpg'), 'Image 1 - 20 Keypoints')
draw_keypoints(img2, keypoints2_limited, os.path.join(output_dir, 'img2_keypoints_limited.jpg'), 'Image 2 - 20 Keypoints')