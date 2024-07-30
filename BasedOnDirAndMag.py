import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class StitchingVideoApp:
    def __init__(self, root, video_path, crop_params=None, plot_interval=10):
        self.root = root
        self.root.title("Video Playback and Stitching")

        # Create GUI elements
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.start_button = ttk.Button(root, text="Start", command=self.start)
        self.start_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        self.canvas = tk.Canvas(root, width=1200, height=600, bg="white")
        self.canvas.grid(row=2, column=0, padx=10, pady=10, columnspan=2)

        self.direction_label = ttk.Label(root, text="Direction: ")
        self.direction_label.grid(row=3, column=0, padx=10, pady=10, columnspan=2)

        self.cap = cv2.VideoCapture(video_path)  # Load the recorded video
        self.running = False
        self.crop_params = crop_params  # Crop parameters: (x, y, width, height)

        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=5, blockSize=7)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.prev_gray = None
        self.p0 = None
        self.frame_count = 0
        self.redetect_interval = 5  # Number of frames between re-detection
        self.plot_interval = plot_interval  # Interval at which we plot frames

        self.canvas_image = np.zeros((600, 1200, 3), np.uint8)  # Empty canvas for stitching
        self.canvas_x = self.canvas_image.shape[1] // 2
        self.canvas_y = self.canvas_image.shape[0] // 2  # Start from the center of the canvas
        self.prev_frame = None

        # Store keypoints and descriptors for the stitched canvas
        self.canvas_keypoints = []
        self.canvas_descriptors = []

        # ORB detector for feature detection and description
        self.orb = cv2.ORB_create()

    def start(self):
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply cropping if crop parameters are defined
                if self.crop_params:
                    x, y, w, h = self.crop_params
                    frame = frame[y:y+h, x:x+w]

                frame = self.resize_image(frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect keypoints and descriptors in the new frame
                kp_new, des_new = self.orb.detectAndCompute(gray, None)

                # Match new frame keypoints with canvas keypoints
                if des_new is not None and len(self.canvas_descriptors) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des_new, np.array(self.canvas_descriptors))
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Use matches to estimate transformation if enough matches are found
                    MIN_MATCH_COUNT = 10
                    if len(matches) > MIN_MATCH_COUNT:
                        src_pts = np.float32([kp_new[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([self.canvas_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        # Warp the new frame using the estimated transformation
                        h, w = frame.shape[:2]
                        aligned_frame = cv2.warpPerspective(frame, M, (w, h))

                        # Stitch the aligned frame onto the canvas
                        self.stitch_and_update_canvas(self.prev_frame, aligned_frame, direction)

                # Re-detect corners periodically or if the number of tracked corners is low
                if self.prev_gray is None or self.frame_count % self.redetect_interval == 0 or (self.p0 is not None and len(self.p0) < 50):
                    self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

                if self.prev_gray is not None and self.p0 is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)

                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        good_old = self.p0[st == 1]

                        # Calculate movement direction
                        if len(good_new) > 0:
                            movement = good_new - good_old
                            median_movement = np.median(movement, axis=0)  # Use median to reduce influence of outliers
                            movement_magnitude = np.linalg.norm(median_movement)

                            if movement_magnitude > 2:  # Threshold for significant movement
                                direction = -median_movement  # Invert direction

                                # Stitch frames if there is significant movement and it's time to plot
                                if self.prev_frame is not None and self.frame_count % self.plot_interval == 0:
                                    self.stitch_and_update_canvas(self.prev_frame, frame, direction)
                                self.prev_frame = frame

                                # Update the direction label
                                direction_text = f"Direction: {self.get_direction_text(direction)}"
                                self.direction_label.config(text=direction_text)

                        self.p0 = good_new.reshape(-1, 1, 2)

                self.prev_gray = gray.copy()
                self.frame_count += 1

                # Convert the frame to a format Tkinter can display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                self.root.after(30, self.update_frame)  # Adjust the delay for desired frame rate

    def resize_image(self, image):
        height, width = image.shape[:2]
        resized = cv2.resize(image, (width // 6, height // 6), interpolation=cv2.INTER_AREA)
        return resized

    def stitch_and_update_canvas(self, image1, image2, direction):
        x_offset = int(direction[0])
        y_offset = int(direction[1])

        self.canvas_x = max(min(self.canvas_x + x_offset, self.canvas_image.shape[1] - image2.shape[1]), 0)
        self.canvas_y = max(min(self.canvas_y + y_offset, self.canvas_image.shape[0] - image2.shape[0]), 0)

        x1, y1 = self.canvas_x, self.canvas_y
        x2, y2 = x1 + image2.shape[1], y1 + image2.shape[0]

        self.canvas_image[y1:y2, x1:x2] = image2

        # Update canvas keypoints and descriptors
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
        if des is not None:
            for k in kp:
                k.pt = (k.pt[0] + self.canvas_x, k.pt[1] + self.canvas_y)
            self.canvas_keypoints.extend(kp)
            self.canvas_descriptors.extend(des)

        self.update_canvas_display()

    def update_canvas_display(self):
        pil_canvas_image = Image.fromarray(self.canvas_image)
        imgtk = ImageTk.PhotoImage(image=pil_canvas_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

    def get_direction_text(self, direction):
        direction_text = ""
        if direction[0] > 0:
            direction_text += "Right "
        elif direction[0] < 0:
            direction_text += "Left "
        if direction[1] > 0:
            direction_text += "Down"
        elif direction[1] < 0:
            direction_text += "Up"
        return direction_text.strip()

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    video_path = "C:\\Users\\humza\\OneDrive\\Desktop\\Auto-Pumping-SickKids\\IMG_1898.mov"  # Replace with the path to your recorded video
    crop_params = (235, 0, 380, 380)  # Replace with (x, y, width, height) to define the crop area
    root = tk.Tk()
    app = StitchingVideoApp(root, video_path, crop_params, 1)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
