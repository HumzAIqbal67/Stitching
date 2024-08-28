import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Label
from PIL import Image, ImageTk

def evaluate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def update_feeds():
    ret, frame = cap.read()
    if ret:
        # Low resolution feed
        low_res_frame = cv2.resize(frame, (320, 240))
        low_res_sharpness = evaluate_sharpness(low_res_frame)
        low_res_img = Image.fromarray(cv2.cvtColor(low_res_frame, cv2.COLOR_BGR2RGB))
        low_res_imgtk = ImageTk.PhotoImage(image=low_res_img)
        low_res_label.imgtk = low_res_imgtk
        low_res_label.config(image=low_res_imgtk)
        low_res_sharpness_label.config(text=f"Low Res Sharpness: {low_res_sharpness:.2f}")

        # High resolution feed with cropping
        h, w, _ = frame.shape
        crop_size = (320, 240)
        x_start = w // 4  # Adjust x_start to change the focus area
        y_start = h // 4  # Adjust y_start to change the focus area
        high_res_frame_cropped = frame[y_start:y_start + crop_size[1], x_start:x_start + crop_size[0]]
        high_res_sharpness = evaluate_sharpness(high_res_frame_cropped)
        high_res_img = Image.fromarray(cv2.cvtColor(high_res_frame_cropped, cv2.COLOR_BGR2RGB))
        high_res_imgtk = ImageTk.PhotoImage(image=high_res_img)
        high_res_label.imgtk = high_res_imgtk
        high_res_label.config(image=high_res_imgtk)
        high_res_sharpness_label.config(text=f"High Res Sharpness: {high_res_sharpness:.2f}")

    # Continuously update the feeds
    low_res_label.after(10, update_feeds)

def set_exposure(val):
    cap.set(cv2.CAP_PROP_EXPOSURE, float(val))

def display_video_from_camera(camera_index=0, width=640, height=480):
    global cap, low_res_label, high_res_label, low_res_sharpness_label, high_res_sharpness_label

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    root = tk.Tk()
    root.title("Dual Feed with Sharpness Evaluation")

    global exposure_slider

    # Create main layout frames
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Left frame for low-resolution feed
    low_res_frame = tk.Frame(main_frame)
    low_res_frame.pack(side=tk.LEFT, padx=10, pady=10)

    low_res_label = Label(low_res_frame)
    low_res_label.pack(pady=5)

    low_res_sharpness_label = Label(low_res_frame, text="Low Res Sharpness: 0.00")
    low_res_sharpness_label.pack(pady=5)

    # Right frame for high-resolution feed
    high_res_frame = tk.Frame(main_frame)
    high_res_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    high_res_label = Label(high_res_frame)
    high_res_label.pack(pady=5)

    high_res_sharpness_label = Label(high_res_frame, text="High Res Sharpness: 0.00")
    high_res_sharpness_label.pack(pady=5)

    # Bottom frame for exposure control
    settings_frame = tk.Frame(root)
    settings_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    exposure_slider = Scale(settings_frame, from_=0, to_=255, orient=tk.HORIZONTAL, label="Exposure")
    exposure_slider.pack(pady=5)
    exposure_slider.set(100)
    exposure_slider.bind("<Motion>", lambda event: set_exposure(exposure_slider.get()))

    # Start updating the feeds
    update_feeds()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera_index = int(input("Enter the camera index: "))
        display_video_from_camera(camera_index, width=640, height=480)
    except ValueError:
        print("Invalid input. Please enter an integer.")
