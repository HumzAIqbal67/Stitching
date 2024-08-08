import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

class StitchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Feed and Image Stitching")
        
        self.video_frame = ttk.Label(self.root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.stitched_frame = ttk.Label(self.root)
        self.stitched_frame.grid(row=0, column=1, padx=10, pady=10)
        
        self.capture_button = ttk.Button(self.root, text="Capture Image", command=self.capture_image)
        self.capture_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.direction_label = ttk.Label(self.root, text="Direction: Unknown")
        self.direction_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.camera_index = 1
        self.width = 320
        self.height = 240
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.images = []
        self.keypoints_and_descriptors = []
        self.translations = []
        self.direction = None
        
        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        self.root.after(10, self.update_video_feed)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            filename = f"capture_{len(self.images)}.jpg"
            cv2.imwrite(filename, frame)
            img = cv2.imread(filename)
            self.images.append((filename, img))
            self.process_image_stitching()

    def process_image_stitching(self):
        sift = cv2.SIFT_create()
        if len(self.images) == 1:
            keypoints, descriptors = sift.detectAndCompute(self.images[0][1], None)
            self.keypoints_and_descriptors.append((keypoints, descriptors))
            self.show_stitched_image(self.images[0][1])
            return
        
        keypoints, descriptors = sift.detectAndCompute(self.images[-1][1], None)
        self.keypoints_and_descriptors.append((keypoints, descriptors))
        
        matches = self.match_descriptors(self.keypoints_and_descriptors[-2][1], self.keypoints_and_descriptors[-1][1])
        translation, filtered_matches = self.compute_translation(matches, self.keypoints_and_descriptors[-2][0], self.keypoints_and_descriptors[-1][0])
        
        if len(self.images) == 2:
            self.direction = self.determine_direction(translation)
            self.direction_label.configure(text=f"Direction: {self.direction}")

        self.translations.append(translation)
        result = self.combine_images(self.images, self.translations)
        self.show_stitched_image(result)

    def match_descriptors(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def compute_translation(self, matches, kp1, kp2):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        angles = np.arctan2(dst_pts[:, 1] - src_pts[:, 1], dst_pts[:, 0] - src_pts[:, 0]) * 180 / np.pi
        vertical_indices = np.where((angles > -90) & (angles < -60))[0]
        
        src_pts = src_pts[vertical_indices]
        dst_pts = dst_pts[vertical_indices]
        filtered_matches = [matches[i] for i in vertical_indices]

        translation = -np.mean(dst_pts - src_pts, axis=0)
        return translation, filtered_matches

    def determine_direction(self, translation):
        if abs(translation[0]) > abs(translation[1]):
            return "Horizontal"
        else:
            return "Vertical"

    def combine_images(self, images, translations):
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

        return canvas

    def show_stitched_image(self, stitched_image):
        stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(stitched_image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.stitched_frame.imgtk = imgtk
        self.stitched_frame.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = StitchingApp(root)
    root.mainloop()
