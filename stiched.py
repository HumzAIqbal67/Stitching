import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class VideoApp:
    def __init__(self, root, video_path, crop_params=None):
        self.root = root
        self.root.title("Video Playback")

        # Create GUI elements
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.start_button = ttk.Button(root, text="Start", command=self.start)
        self.start_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        self.canvas_label = ttk.Label(root)
        self.canvas_label.grid(row=0, column=2, padx=10, pady=10, columnspan=2)

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
        self.plot_interval = 30  # Number of frames between plotting the image onto the canvas

        # Initialize the canvas
        self.canvas_size_multiplier = 5  # Smaller canvas
        self.image_scale = 0.25  # Scale the image to a fourth of the size
        self.canvas = None
        self.canvas_initialized = False

        # Initial position on canvas (close to top-left, but not exactly)
        self.initial_offset = 50
        self.canvas_x = self.initial_offset
        self.canvas_y = self.initial_offset

        self.last_frame = None
        self.last_direction = None

    def start(self):
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False

    def initialize_canvas(self, frame):
        height, width, _ = frame.shape
        self.canvas_height = int(height * self.canvas_size_multiplier)
        self.canvas_width = int(width * self.canvas_size_multiplier)
        self.canvas = np.full((self.canvas_height, self.canvas_width, 3), 128, dtype=np.uint8)  # Grey background
        self.canvas_initialized = True

        # Place the first frame close to the top-left of the canvas
        self.canvas[self.canvas_y:self.canvas_y+height, self.canvas_x:self.canvas_x+width] = frame

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply cropping if crop parameters are defined
                if self.crop_params:
                    x, y, w, h = self.crop_params
                    frame = frame[y:y+h, x:x+w]

                # Resize the frame to a fourth of its size
                frame = cv2.resize(frame, (0, 0), fx=self.image_scale, fy=self.image_scale)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if not self.canvas_initialized:
                    self.initialize_canvas(frame)

                # Re-detect corners periodically or if the number of tracked corners is low
                if self.prev_gray is None or self.frame_count % self.redetect_interval == 0 or len(self.p0) < 50:
                    self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

                if self.prev_gray is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)

                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        good_old = self.p0[st == 1]

                        # Calculate movement direction
                        if len(good_new) > 0:
                            movement = good_new - good_old
                            avg_movement = np.mean(movement, axis=0)
                            movement_magnitude = np.linalg.norm(avg_movement)

                            direction = None
                            if movement_magnitude > 2:  # Threshold for significant movement
                                direction_angle = np.arctan2(avg_movement[1], avg_movement[0]) * 180 / np.pi

                                # Determine direction
                                if -45 <= direction_angle < 45:
                                    direction = 'Right'
                                elif 45 <= direction_angle < 135:
                                    direction = 'Up'
                                elif direction_angle >= 135 or direction_angle < -135:
                                    direction = 'Left'
                                elif -135 <= direction_angle < -45:
                                    direction = 'Down'

                                # Draw arrows indicating direction (only on video, not on canvas)
                                for i, (new, old) in enumerate(zip(good_new, good_old)):
                                    a, b = new.ravel()
                                    c, d = old.ravel()
                                    frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

                                # Overlay direction arrows and labels (only on video, not on canvas)
                                frame = self.overlay_direction_arrows(frame, direction)

                                # Plot the frame onto the canvas periodically
                                if self.frame_count % self.plot_interval == 0:
                                    if self.last_frame is not None and self.last_direction is not None:
                                        self.stitch_and_plot(frame, self.last_frame, self.last_direction)
                                    self.last_frame = frame.copy()
                                    self.last_direction = direction

                        self.p0 = good_new.reshape(-1, 1, 2)

                self.prev_gray = gray.copy()
                self.frame_count += 1

                # Convert the frame to a format Tkinter can display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Convert the canvas to a format Tkinter can display
                canvas_img = Image.fromarray(self.canvas)
                canvas_imgtk = ImageTk.PhotoImage(image=canvas_img)
                self.canvas_label.imgtk = canvas_imgtk
                self.canvas_label.configure(image=canvas_imgtk)

                self.root.after(30, self.update_frame)  # Adjust the delay for desired frame rate

    def stitch_and_plot(self, frame1, frame2, direction):
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

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
            return  # Not enough matches to compute homography

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate homography
        h_matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp the second frame to align with the first frame
        height, width, _ = frame1.shape
        stitched_image = cv2.warpPerspective(frame2, h_matrix, (width * 2, height * 2))

        # Determine the offset based on direction
        if direction == 'Right':
            x_offset = self.canvas_x + width // 2
            y_offset = self.canvas_y
        elif direction == 'Left':
            x_offset = self.canvas_x - width // 2
            y_offset = self.canvas_y
        elif direction == 'Up':
            x_offset = self.canvas_x
            y_offset = self.canvas_y - height // 2
        elif direction == 'Down':
            x_offset = self.canvas_x
            y_offset = self.canvas_y + height // 2
        else:
            return

        # Update canvas position
        self.canvas_x = x_offset
        self.canvas_y = y_offset

        # Blend the stitched image into the canvas
        canvas_height, canvas_width, _ = self.canvas.shape
        blend_height = min(stitched_image.shape[0], canvas_height - y_offset)
        blend_width = min(stitched_image.shape[1], canvas_width - x_offset)

        self.canvas[y_offset:y_offset+blend_height, x_offset:x_offset+blend_width] = stitched_image[:blend_height, :blend_width]

    def overlay_direction_arrows(self, frame, active_direction):
        # Create an overlay for arrows
        arrow_length = 10  # Smaller arrows
        thickness = 2
        color_inactive = (0, 0, 255)
        color_active = (0, 255, 0)
        
        height, width, _ = frame.shape
        top_left_x, top_left_y = 20, 20  # Adjusted position to fully display arrows

        directions = {
            'Up': ((top_left_x, top_left_y), (top_left_x, top_left_y - arrow_length)),
            'Down': ((top_left_x, top_left_y), (top_left_x, top_left_y + arrow_length)),
            'Left': ((top_left_x, top_left_y), (top_left_x - arrow_length, top_left_y)),
            'Right': ((top_left_x, top_left_y), (top_left_x + arrow_length, top_left_y))
        }

        for direction, ((x1, y1), (x2, y2)) in directions.items():
            color = color_active if direction == active_direction else color_inactive
            frame = cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, thickness)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1

        labels = {
            'Up': (top_left_x - 10, top_left_y - arrow_length - 5),
            'Down': (top_left_x - 10, top_left_y + arrow_length + 10),
            'Left': (top_left_x - arrow_length - 30, top_left_y + 5),
            'Right': (top_left_x + arrow_length + 5, top_left_y + 5)
        }

        for direction, (text_x, text_y) in labels.items():
            color = color_active if direction == active_direction else color_inactive
            frame = cv2.putText(frame, direction, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        return frame

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    video_path = "C:\\Users\\humza\\OneDrive\\Desktop\\Auto-Pumping-SickKids\\IMG_1898.mov"  # Replace with the path to your recorded video
    crop_params = (225, 0, 390, 400)  # Replace with (x, y, width, height) to define the crop area
    root = tk.Tk()
    app = VideoApp(root, video_path, crop_params)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
