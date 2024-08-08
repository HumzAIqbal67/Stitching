import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Feed and Stitching")

        # Create a directory to save captured images
        self.image_folder = "captured_images"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        self.image_counter = 0

        # Create a frame for the live feed and controls
        self.frame_left = tk.Frame(root)
        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a label to hold the video feed
        self.video_label = tk.Label(self.frame_left)
        self.video_label.pack()

        # Create a capture button
        self.capture_button = tk.Button(self.frame_left, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=10)

        # Create a stitch button
        self.stitch_button = tk.Button(self.frame_left, text="Stitch", command=self.stitch_images)
        self.stitch_button.pack(pady=10)

        # Create a frame for the captured images hamper
        self.frame_right = tk.Frame(root)
        self.frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.scroll_canvas = tk.Canvas(self.frame_right, width=150)
        self.scroll_frame = tk.Frame(self.scroll_canvas)
        self.scrollbar = tk.Scrollbar(self.frame_right, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT)
        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        self.scroll_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.captured_images = []

        # Open the video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            self.root.quit()

        # Start the update loop
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize the frame (optional)
            frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to an ImageTk format
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            # Update the label with the new image
            self.video_label.config(image=photo)
            self.video_label.image = photo

        # Call the update_video method again after 30 ms
        self.root.after(30, self.update_video)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            filename = os.path.join(self.image_folder, f"image_{self.image_counter:03d}.jpg")
            cv2.imwrite(filename, frame)
            self.image_counter += 1
            self.display_captured_image(filename)

    def display_captured_image(self, filepath):
        image = Image.open(filepath)
        image.thumbnail((100, 100))  # Resize for the hamper
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self.scroll_frame, image=photo)
        label.image = photo
        label.pack(pady=2)
        self.captured_images.append(filepath)

    def stitch_images(self):
        if len(self.captured_images) < 2:
            print("Not enough images to stitch!")
            return

        images = self.load_images(self.captured_images)
        stitched_image = self.stitch(images)
        if stitched_image is not None:
            cv2.imwrite("stitched_image.jpg", stitched_image)
            cv2.imshow("Stitched Image", stitched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def load_images(self, filepaths):
        images = []
        for filepath in filepaths:
            img = cv2.imread(filepath)
            if img is not None:
                images.append(img)
        return images

    def stitch(self, images):
        sift = cv2.SIFT_create()

        keypoints_and_descriptors = []
        for img in images:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            keypoints_and_descriptors.append((keypoints, descriptors))

        matches_list = []
        translations = []
        for i in range(len(images) - 1):
            matches = self.match_descriptors(keypoints_and_descriptors[i][1], keypoints_and_descriptors[i + 1][1])
            translation, filtered_matches = self.compute_translation(matches, keypoints_and_descriptors[i][0], keypoints_and_descriptors[i + 1][0])
            matches_list.append(filtered_matches)
            translations.append(translation)

        table_image = self.draw_matches(images, keypoints_and_descriptors, matches_list)
        cv2.imshow("Keypoints and Matches Table", table_image)
        cv2.waitKey(0)

        result = self.combine_images(images, translations)
        return result

    def match_descriptors(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def compute_translation(self, matches, kp1, kp2):
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

    def draw_matches(self, images, keypoints_and_descriptors, matches_list):
        num_images = len(images)
        height, width, _ = images[0].shape

        table_image = np.zeros((height * num_images, width * 3, 3), dtype=np.uint8)

        for i in range(num_images):
            img = images[i]
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

        return table_image

    def combine_images(self, images, translations):
        height, width, _ = images[0].shape
        total_translation = np.cumsum(translations, axis=0)
        max_translation = np.max(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)
        min_translation = np.min(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)

        canvas_width = width
        canvas_height = height + (max_translation[1] - min_translation[1])

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        y_offset = -min_translation[1]
        canvas[y_offset:y_offset + height, 0:width] = images[0]

        current_y = y_offset

        for i in range(1, len(images)):
            current_y += translations[i - 1][1]
            y_offset_int = int(current_y)
            canvas[y_offset_int:y_offset_int + height, 0:width] = images[i]
            cv2.imshow(f"Stitched Progress {i}", canvas)
            cv2.waitKey(0)  # Wait for a key press to view each step

        return canvas

    def __del__(self):
        # Release the capture when the application is closed
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()