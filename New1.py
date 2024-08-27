import cv2
import tkinter as tk
from tkinter import filedialog
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

        translations = []
        for i in range(len(images) - 1):
            matches = self.match_descriptors(keypoints_and_descriptors[i][1], keypoints_and_descriptors[i + 1][1])
            translation, _ = self.compute_translation(matches, keypoints_and_descriptors[i][0], keypoints_and_descriptors[i + 1][0])
            translations.append(translation)

        # Compute the median translation vector
        translations = np.array(translations)
        median_translation = np.median(translations, axis=0)

        # Calculate the distance from each translation to the median
        distances = np.linalg.norm(translations - median_translation, axis=1)

        # Filter out translations that are too far from the median
        threshold = 1.5 * np.median(distances)
        valid_indices = distances < threshold
        translations = translations[valid_indices]

        # Stitch the images with valid translations
        current_x, current_y = 0, 0
        max_x, max_y = images[0].shape[1], images[0].shape[0]
        min_x, min_y = 0, 0

        for i in range(len(translations)):
            current_x -= translations[i][0]
            current_y -= translations[i][1]
            max_x = max(max_x, current_x + images[i+1].shape[1])
            max_y = max(max_y, current_y + images[i+1].shape[0])
            min_x = min(min_x, current_x)
            min_y = min(min_y, current_y)

        # Ensure canvas size includes the maximum width and height considering all translations
        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)

        # Create a gray canvas
        canvas = np.full((canvas_height, canvas_width, 3), 128, dtype=np.uint8)

        x_offset = int(-min_x)
        y_offset = int(-min_y)
        canvas[y_offset:y_offset + images[0].shape[0], x_offset:x_offset + images[0].shape[1]] = images[0]

        current_x, current_y = 0, 0
        for i in range(len(translations)):
            current_x -= translations[i][0]
            current_y -= translations[i][1]
            x_offset_int = int(current_x - min_x)
            y_offset_int = int(current_y - min_y)
            canvas[y_offset_int:y_offset_int + images[i+1].shape[0], x_offset_int:x_offset_int + images[i+1].shape[1]] = images[i+1]

        return canvas

    def match_descriptors(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def compute_translation(self, matches, kp1, kp2):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        translation = np.mean(dst_pts - src_pts, axis=0)
        return translation, matches

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
