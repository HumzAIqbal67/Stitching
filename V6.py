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

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                                    direction = 'Left'
                                elif 45 <= direction_angle < 135:
                                    direction = 'Up'
                                elif direction_angle >= 135 or direction_angle < -135:
                                    direction = 'Right'
                                elif -135 <= direction_angle < -45:
                                    direction = 'Down'

                                # Draw arrows indicating direction
                                for i, (new, old) in enumerate(zip(good_new, good_old)):
                                    a, b = new.ravel()
                                    c, d = old.ravel()
                                    frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

                            # Overlay direction arrows and labels
                            frame = self.overlay_direction_arrows(frame, direction)

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

    def overlay_direction_arrows(self, frame, active_direction):
        # Create an overlay for arrows
        arrow_length = 50
        thickness = 5
        color_inactive = (0, 0, 255)
        color_active = (0, 255, 0)
        
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        directions = {
            'Up': ((center_x, center_y), (center_x, center_y - arrow_length)),
            'Down': ((center_x, center_y), (center_x, center_y + arrow_length)),
            'Left': ((center_x, center_y), (center_x - arrow_length, center_y)),
            'Right': ((center_x, center_y), (center_x + arrow_length, center_y))
        }

        for direction, ((x1, y1), (x2, y2)) in directions.items():
            color = color_active if direction == active_direction else color_inactive
            frame = cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, thickness)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        labels = {
            'Up': (center_x - 20, center_y - arrow_length - 10),
            'Down': (center_x - 30, center_y + arrow_length + 20),
            'Left': (center_x - arrow_length - 50, center_y + 5),
            'Right': (center_x + arrow_length + 10, center_y + 5)
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
