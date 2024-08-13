import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import numpy as np

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Image Stitching")

        # Create a frame for the controls
        self.frame_left = tk.Frame(root)
        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)

        # Create an entry to input the folder path
        self.folder_label = tk.Label(self.frame_left, text="Image Folder:")
        self.folder_label.pack(pady=5)

        self.folder_entry = tk.Entry(self.frame_left, width=50)
        self.folder_entry.pack(pady=5)

        self.browse_button = tk.Button(self.frame_left, text="Browse", command=self.browse_folder)
        self.browse_button.pack(pady=5)

        # Create a stitch button
        self.stitch_button = tk.Button(self.frame_left, text="Stitch Images", command=self.stitch_images)
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

        self.image_folder = None
        self.captured_images = []
        self.stitching_direction = None
        self.is_positive_direction = None

    def browse_folder(self):
        self.image_folder = filedialog.askdirectory()
        if self.image_folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, self.image_folder)
            self.load_folder_images()

    def load_folder_images(self):
        self.captured_images = []
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if self.image_folder and os.path.exists(self.image_folder):
            files = sorted(os.listdir(self.image_folder))
            for file in files:
                filepath = os.path.join(self.image_folder, file)
                if os.path.isfile(filepath):
                    self.captured_images.append(filepath)
                    self.display_captured_image(filepath)

    def display_captured_image(self, filepath):
        image = Image.open(filepath)
        image.thumbnail((100, 100))  # Resize for the hamper
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self.scroll_frame, image=photo)
        label.image = photo
        label.pack(pady=2)

    def stitch_images(self):
        if len(self.captured_images) < 2:
            print("Not enough images to stitch!")
            return

        images = self.load_images(self.captured_images)
        self.determine_stitching_direction(images[0], images[1])
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

    def determine_stitching_direction(self, img1, img2):
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        
        matches = self.match_descriptors(descriptors1, descriptors2)
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        translation_vector = np.mean(dst_pts - src_pts, axis=0)
        
        if abs(translation_vector[0]) > abs(translation_vector[1]):
            self.stitching_direction = 'horizontal'
            self.is_positive_direction = translation_vector[0] > 0
            print(f"Stitching direction determined: {'Left to Right' if self.is_positive_direction else 'Right to Left'}")
        else:
            self.stitching_direction = 'vertical'
            self.is_positive_direction = translation_vector[1] > 0
            print(f"Stitching direction determined: {'Top to Bottom' if self.is_positive_direction else 'Bottom to Top'}")

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

        if self.stitching_direction == 'horizontal':
            if self.is_positive_direction:
                # Positive horizontal movement (left to right)
                valid_indices = np.where((angles > -30) & (angles < 30))[0]
            else:
                # Negative horizontal movement (right to left)
                valid_indices = np.where((angles > 150) | (angles < -150))[0]
        else:
            if self.is_positive_direction:
                # Positive vertical movement (top to bottom)
                valid_indices = np.where((angles > 60) & (angles < 120))[0]
            else:
                # Negative vertical movement (bottom to top)
                valid_indices = np.where((angles > -120) & (angles < -60))[0]

        # Filter the points and matches based on the determined direction
        src_pts = src_pts[valid_indices]
        dst_pts = dst_pts[valid_indices]
        filtered_matches = [matches[i] for i in valid_indices]

        translation = -np.mean(dst_pts - src_pts, axis=0)
        return translation, filtered_matches

    def combine_images(self, images, translations):
        height, width, _ = images[0].shape
        total_translation = np.cumsum(translations, axis=0)
        max_translation = np.max(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)
        min_translation = np.min(np.vstack((total_translation, [[0, 0]])), axis=0).astype(int)

        if self.stitching_direction == 'horizontal':
            canvas_width = width + (max_translation[0] - min_translation[0])
            canvas_height = height
        else:
            canvas_width = width
            canvas_height = height + (max_translation[1] - min_translation[1])

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        x_offset = -min_translation[0]
        y_offset = -min_translation[1]
        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = images[0]

        current_x = x_offset
        current_y = y_offset

        for i in range(1, len(images)):
            if self.stitching_direction == 'horizontal':
                current_x += translations[i - 1][0]
                x_offset_int = int(current_x)
                canvas[y_offset:y_offset + height, x_offset_int:x_offset_int + width] = images[i]
            else:
                current_y += translations[i - 1][1]
                y_offset_int = int(current_y)
                canvas[y_offset_int:y_offset_int + height, x_offset:x_offset + width] = images[i]

        return canvas

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
