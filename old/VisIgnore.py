import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_translation(matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Calculate the angle of each vector
    angles = np.arctan2(dst_pts[:, 1] - src_pts[:, 1], dst_pts[:, 0] - src_pts[:, 0]) * 180 / np.pi
    vertical_indices = np.where((angles > -90) & (angles < -60))[0]  # Adjust the angle threshold as needed

    # Filter the points and matches based on the vertical direction
    src_pts = src_pts[vertical_indices]
    dst_pts = dst_pts[vertical_indices]
    filtered_matches = [matches[i] for i in vertical_indices]

    translation = -np.mean(dst_pts - src_pts, axis=0)
    return translation, filtered_matches

def draw_matches(images, keypoints_and_descriptors, matches_list):
    num_images = len(images)
    height, width, _ = images[0][1].shape

    table_image = np.zeros((height * num_images, width * 3, 3), dtype=np.uint8)

    for i in range(num_images):
        filename, img = images[i]
        kp = keypoints_and_descriptors[i][0]

        img_with_kp = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        table_image[i * height:(i + 1) * height, width:2 * width] = img_with_kp

        if i < num_images - 1:
            kp2 = keypoints_and_descriptors[i + 1][0]
            matches = matches_list[i]

            img_matches = np.copy(img)
            for match in matches:
                pt1 = (int(kp[match.queryIdx].pt[0]), int(kp[match.queryIdx].pt[1]))
                pt2 = (int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1]))
                cv2.circle(img_matches, pt1, 5, (255, 0, 0), -1)
                cv2.line(img_matches, pt1, pt2, (0, 255, 0), 2)

            table_image[i * height:(i + 1) * height, 2 * width:] = img_matches

        cv2.putText(table_image, filename, (10, i * height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return table_image

def combine_images(images, translations):
    height, width, _ = images[0][1].shape
    total_translation = np.cumsum(translations, axis=0)
    max_translation = np.max(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)
    min_translation = np.min(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)

    canvas_width = width
    canvas_height = height + (max_translation[1] - min_translation[1])

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    y_offset = -min_translation[1]
    canvas[y_offset:y_offset + height, 0:width] = images[0][1]

    current_y = y_offset

    for i in range(1, len(images)):
        current_y += translations[i - 1][1]
        y_offset_int = int(current_y)
        canvas[y_offset_int:y_offset_int + height, 0:width] = images[i][1]
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
    for _, img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_and_descriptors.append((keypoints, descriptors))

    matches_list = []
    translations = []
    for i in range(len(images) - 1):
        matches = match_descriptors(keypoints_and_descriptors[i][1], keypoints_and_descriptors[i + 1][1])
        translation, filtered_matches = compute_translation(matches, keypoints_and_descriptors[i][0], keypoints_and_descriptors[i + 1][0])
        matches_list.append(filtered_matches)
        translations.append(translation)

    table_image = draw_matches(images, keypoints_and_descriptors, matches_list)
    cv2.imshow("Keypoints and Matches Table", table_image)
    cv2.waitKey(0)

    result = combine_images(images, translations)

    cv2.imwrite("stitched_image.jpg", result)
    cv2.imshow("Stitched Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

