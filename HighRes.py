# Progress bar and high res pictures - no ignoring black lines.

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import Scale, Label, Entry, Button, Toplevel, ttk
from PIL import Image, ImageTk

def evaluate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def stitch_images(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use SIFT to detect and compute key points and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract locations of matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the affine transformation matrix (translation only)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Flip the translation (if needed)
    M[0, 2] = -M[0, 2]  # Flip the translation along the x-axis
    M[1, 2] = -M[1, 2]  # Flip the translation along the y-axis

    # Calculate the size of the stitched image canvas
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Calculate the translation offsets
    translation_x = M[0, 2]
    translation_y = M[1, 2]

    # Determine canvas size and placement based on translation direction
    if translation_x >= 0 and translation_y >= 0:
        stitched_w = max(w1, w2 + int(translation_x))
        stitched_h = max(h1, h2 + int(translation_y))
        x_offset, y_offset = 0, 0
    elif translation_x < 0 and translation_y >= 0:
        stitched_w = max(w1 - int(translation_x), w2)
        stitched_h = max(h1, h2 + int(translation_y))
        x_offset, y_offset = abs(int(translation_x)), 0
    elif translation_x >= 0 and translation_y < 0:
        stitched_w = max(w1, w2 + int(translation_x))
        stitched_h = max(h1 - int(translation_y), h2)
        x_offset, y_offset = 0, abs(int(translation_y))
    else:
        stitched_w = max(w1 - int(translation_x), w2)
        stitched_h = max(h1 - int(translation_y), h2)
        x_offset, y_offset = abs(int(translation_x)), abs(int(translation_y))

    # Create a canvas large enough to hold both images
    stitched_image = np.ones((stitched_h, stitched_w, 3), dtype=np.uint8) * 255

    # Place the first image on the canvas
    stitched_image[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1

    # Adjust the translation matrix to correctly align the second image
    M[0, 2] += x_offset
    M[1, 2] += y_offset

    # Warp the second image using the corrected transformation matrix
    img2_aligned = cv2.warpAffine(img2, M, (stitched_image.shape[1], stitched_image.shape[0]))

    # Combine the images by overlaying the aligned second image
    img2_aligned_gray = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    mask = img2_aligned_gray > 0
    stitched_image[mask] = img2_aligned[mask]

    return stitched_image

def save_image(image, name, resolution="original"):
    directory = "photos"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f"{name}_{resolution}.jpg")
    cv2.imwrite(filename, image)

def capture_and_stitch():
    global first_capture, first_capture_high_res, img_counter

    ret, frame = cap.read()
    print(np.shape(frame))
    if ret:
        # Save the original high-resolution image
        save_image(frame, f"image_{img_counter}", resolution="high")

        # Resize to low-resolution for live stitching
        frame_resized = cv2.resize(frame, (320, 240))
        save_image(frame_resized, f"image_{img_counter}", resolution="low")

        sharpness = evaluate_sharpness(frame_resized)

        if sharpness >= float(sharpness_threshold.get()):
            if first_capture is None:
                first_capture = frame_resized
                first_capture_high_res = frame
            else:
                second_capture = frame_resized
                second_capture_high_res = frame

                # Stitch low-resolution images for live feed
                stitched_image = stitch_images(first_capture, second_capture)
                save_image(stitched_image, f"stitched_{img_counter}", resolution="low")

                # Display stitched image on the canvas
                img = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                canvas.create_image(400, 300, anchor=tk.CENTER, image=imgtk)
                canvas.image = imgtk

                # Prepare for high-resolution stitching in the background
                start_high_res_stitching(first_capture_high_res, second_capture_high_res)

                # Update the first_capture to the stitched image for the next iteration
                first_capture = stitched_image
                first_capture_high_res = stitched_image

            img_counter += 1

def start_high_res_stitching(img1, img2):
    progress_window = Toplevel(root)
    progress_window.title("High-Resolution Stitching")

    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", mode="determinate", length=300)
    progress_bar.pack(pady=20)
    progress_bar.start()

    def high_res_stitch():
        # Perform the high-resolution stitching
        stitched_image_high_res = stitch_images(img1, img2)
        save_image(stitched_image_high_res, f"stitched_{img_counter - 1}", resolution="high")

        # Close the progress bar window when done
        progress_window.destroy()

    root.after(100, high_res_stitch)

def update_feed():
    ret, frame = cap.read()
    if ret:
        frame_resized = cv2.resize(frame, (320, 240))
        sharpness = evaluate_sharpness(frame_resized)
        sharpness_label.config(text=f"Sharpness: {sharpness:.2f}")
        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        live_feed_label.imgtk = imgtk
        live_feed_label.config(image=imgtk)

    live_feed_label.after(10, update_feed)

def set_exposure(val):
    cap.set(cv2.CAP_PROP_EXPOSURE, float(val))

def display_video_from_camera(camera_index=0, width=640, height=480):
    global cap, first_capture, first_capture_high_res, img_counter
    first_capture = None
    first_capture_high_res = None
    img_counter = 0

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize the GUI
    global root, canvas, live_feed_label, sharpness_label, sharpness_threshold, exposure_slider
    root = tk.Tk()
    root.title("Microscope Image Stitching")

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame, width=800, height=600)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    settings_frame = tk.Frame(main_frame)
    settings_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    live_feed_label = Label(settings_frame)
    live_feed_label.pack(pady=5)

    sharpness_label = Label(settings_frame, text="Sharpness: 0.00")
    sharpness_label.pack(pady=5)

    Label(settings_frame, text="Sharpness Threshold:").pack(pady=5)
    sharpness_threshold = Entry(settings_frame)
    sharpness_threshold.pack(pady=5)
    sharpness_threshold.insert(0, "100.0")

    exposure_slider = Scale(settings_frame, from_=0, to_=255, orient=tk.HORIZONTAL, label="Exposure")
    exposure_slider.pack(pady=5)
    exposure_slider.set(100)
    exposure_slider.bind("<Motion>", lambda event: set_exposure(exposure_slider.get()))

    capture_button = Button(settings_frame, text="Capture and Stitch", command=capture_and_stitch)
    capture_button.pack(pady=5)

    update_feed()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera_index = int(input("Enter the camera index: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
    else:
        display_video_from_camera(camera_index, width=320, height=240)
