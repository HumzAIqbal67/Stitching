import cv2
import numpy as np

class StitchingApp:
    def __init__(self, image1_path, image2_path, output_path):
        # Load the images
        self.image1 = cv2.imread(image1_path)
        self.image2 = cv2.imread(image2_path)

        # Check if images are loaded
        if self.image1 is None or self.image2 is None:
            raise ValueError("One or both image paths are incorrect or the images are not accessible")

        # Resize images to ensure they fit on the screen
        self.image1 = self.resize_image(self.image1, width=400)
        self.image2 = self.resize_image(self.image2, width=400)

        # Convert images to grayscale
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        # Detect features and compute descriptors using SIFT
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Find matches between descriptors using brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        if len(matches) < 4:
            raise ValueError("Not enough matches found to compute homography")

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate homography
        h_matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp the second image to align with the first image
        height, width, _ = self.image1.shape
        self.stitched_image = cv2.warpPerspective(self.image2, h_matrix, (width, height + self.image2.shape[0]))
        self.stitched_image[0:height, 0:width] = self.image1

        # Save the stitched image
        self.save_image(output_path)

    def resize_image(self, image, width=None, height=None):
        if width is None and height is None:
            return image

        (h, w) = image.shape[:2]
        if width is not None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            r = height / float(h)
            dim = (int(w * r), height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    def save_image(self, output_path):
        cv2.imwrite(output_path, self.stitched_image)

if __name__ == "__main__":
    image1_path = "C:\\Users\\humza\\OneDrive\\Desktop\\Auto-Pumping-SickKids\\image.png"  # Replace with the path to your first image
    image2_path = "C:\\Users\\humza\\OneDrive\\Desktop\\Auto-Pumping-SickKids\\image1.png"  # Replace with the path to your second image
    output_path = "stitched_image.png"  # Replace with the desired output path for the stitched image
    app = StitchingApp(image1_path, image2_path, output_path)


