import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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

        # Open the video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            self.root.quit()

        # Create sliders for exposure, exposure time, and gain
        self.exposure_slider = tk.Scale(self.frame_left, from_=-13, to_=0, orient=tk.HORIZONTAL, label="Exposure", command=self.update_exposure)
        self.exposure_slider.pack(pady=10)

        self.exposure_time_slider = tk.Scale(self.frame_left, from_=1, to_=10000, orient=tk.HORIZONTAL, label="Exposure Time (µs)", command=self.update_exposure_time)
        self.exposure_time_slider.pack(pady=10)

        self.gain_slider = tk.Scale(self.frame_left, from_=0, to_=255, orient=tk.HORIZONTAL, label="Gain", command=self.update_gain)
        self.gain_slider.pack(pady=10)

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

    def update_exposure(self, value):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def update_exposure_time(self, value):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value) / 10000)

    def update_gain(self, value):
        self.cap.set(cv2.CAP_PROP_GAIN, float(value))

    def __del__(self):
        # Release the capture when the application is closed
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraSettingsApp(root)
    root.mainloop()
