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
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.prev_gray = None
        self.p0 = None

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

                if self.prev_gray is None:
                    self.prev_gray = gray
                    self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                else:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
                    good_new = p1[st == 1]
                    good_old = self.p0[st == 1]

                    # Calculate movement direction
                    if len(good_new) > 0:
                        movement = good_new - good_old
                        avg_movement = np.mean(movement, axis=0)
                        direction = np.arctan2(avg_movement[1], avg_movement[0]) * 180 / np.pi
                        direction_text = f"Direction: {direction:.2f}°"

                        # Draw arrows indicating direction
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

                        cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    self.prev_gray = gray.copy()
                    self.p0 = good_new.reshape(-1, 1, 2)

                # Convert the frame to a format Tkinter can display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                self.root.after(30, self.update_frame)  # Adjust the delay for desired frame rate

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
