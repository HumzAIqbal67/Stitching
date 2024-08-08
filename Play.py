import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

class CameraSettingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Settings")

        # Create a frame for the live feed and controls
        self.frame_left = tk.Frame(root)
        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a label to hold the video feed
        self.video_label = tk.Label(self.frame_left)
        self.video_label.pack()

        # Create a label to display the FPS
        self.fps_label = tk.Label(self.frame_left, text="FPS: 0")
        self.fps_label.pack(pady=5)

        # Open the video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            self.root.quit()

        # Create sliders and entries for exposure, exposure time, and gain
        self.exposure_slider = tk.Scale(self.frame_left, from_=-10, to_=10, orient=tk.HORIZONTAL, label="Exposure (EV units)", command=self.update_exposure)
        self.exposure_slider.pack(pady=10)

        self.exposure_time_label = tk.Label(self.frame_left, text="Exposure Time (ms)")
        self.exposure_time_label.pack(pady=5)
        self.exposure_time_entry = tk.Entry(self.frame_left)
        self.exposure_time_entry.pack(pady=5)
        self.exposure_time_entry.bind("<Return>", self.update_exposure_time)

        self.gain_label = tk.Label(self.frame_left, text="Gain")
        self.gain_label.pack(pady=5)
        self.gain_entry = tk.Entry(self.frame_left)
        self.gain_entry.pack(pady=5)
        self.gain_entry.bind("<Return>", self.update_gain)

        # Start the update loop
        self.start_time = time.time()
        self.frame_count = 0
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

            # Update FPS calculation
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                fps = self.frame_count / elapsed_time
                self.fps_label.config(text=f"FPS: {fps:.2f}")
                self.start_time = time.time()
                self.frame_count = 0

        # Call the update_video method again after 30 ms
        self.root.after(30, self.update_video)

    def update_exposure(self, value):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def update_exposure_time(self, event):
        value = self.exposure_time_entry.get()
        try:
            exposure_time = float(value)
            if 1 <= exposure_time <= 1000:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_time / 1000)  # Assuming the entry is in milliseconds
            else:
                print("Exposure time out of range (1-1000 ms)")
        except ValueError:
            print("Invalid exposure time value")

    def update_gain(self, event):
        value = self.gain_entry.get()
        try:
            gain = float(value)
            if 0 <= gain <= 255:
                self.cap.set(cv2.CAP_PROP_GAIN, gain)
            else:
                print("Gain out of range (0-255)")
        except ValueError:
            print("Invalid gain value")

    def __del__(self):
        # Release the capture when the application is closed
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraSettingsApp(root)
    root.mainloop()


