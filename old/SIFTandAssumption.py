import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_translation(matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    translation = np.mean(src_pts - dst_pts, axis=0)
    return translation

def combine_images(images, translations):
    height, width, _ = images[0].shape
    total_translation = np.cumsum(translations, axis=0)
    max_translation = np.max(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)
    min_translation = np.min(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)

    canvas_width = width + (max_translation[0] - min_translation[0])
    canvas_height = height + (max_translation[1] - min_translation[1])

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    x_offset, y_offset = -min_translation
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = images[0]

    current_x, current_y = x_offset, y_offset

    for i in range(1, len(images)):
        current_x += translations[i - 1][0]
        current_y += translations[i - 1][1]
        x_offset_int = int(current_x)
        y_offset_int = int(current_y)
        
        # Calculate where to place the current image
        new_height, new_width = images[i].shape[:2]
        y1, y2 = max(0, y_offset_int), min(y_offset_int + new_height, canvas_height)
        x1, x2 = max(0, x_offset_int), min(x_offset_int + new_width, canvas_width)
        
        canvas[y1:y2, x1:x2] = images[i][y1 - y_offset_int:y2 - y_offset_int, x1 - x_offset_int:x2 - x_offset_int]
        cv2.imshow(f"Stitched Progress {i}", canvas)
        cv2.waitKey(0)  # Wait for a key press to view each step

    return canvas

def main():
    folder = "images"
    images = load_images_from_folder(folder)
    if len(images) == 0:
        print("No images to stitch!")
        return

    sift = cv2.SIFT_create()

    keypoints_and_descriptors = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_and_descriptors.append((keypoints, descriptors))

    matches_list = []
    for i in range(len(images) - 1):
        matches = match_descriptors(keypoints_and_descriptors[i][1], keypoints_and_descriptors[i + 1][1])
        matches_list.append(matches)

    translations = []
    for i in range(len(matches_list)):
        translation = compute_translation(matches_list[i], keypoints_and_descriptors[i][0], keypoints_and_descriptors[i + 1][0])
        translations.append(translation)

    result = combine_images(images, translations)

    cv2.imwrite("stitched_image.jpg", result)
    cv2.imshow("Stitched Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
