import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Label, Entry, Button
from PIL import Image, ImageTk

def evaluate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def stitch_images(stitched_image, new_image):
    # Convert images to grayscale
    stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Mask out black areas in the stitched image
    mask = stitched_gray > 0
    stitched_gray_masked = cv2.bitwise_and(stitched_gray, stitched_gray, mask=mask.astype(np.uint8))

    # Use SIFT to detect and compute key points and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(stitched_gray_masked, None)
    kp2, des2 = sift.detectAndCompute(new_gray, None)

    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract locations of matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the affine transformation matrix (translation only)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Calculate the size of the stitched image canvas
    h1, w1, _ = stitched_image.shape
    h2, w2, _ = new_image.shape

    # Calculate the translation offsets
    translation_x = M[0, 2]
    translation_y = M[1, 2]

    # Calculate the new size for the stitched image
    stitched_w = max(w1, w2 + int(translation_x))
    stitched_h = max(h1, h2 + int(translation_y))

    # Create a canvas large enough to hold both images
    new_stitched_image = np.zeros((stitched_h, stitched_w, 3), dtype=np.uint8)

    # Place the first image on the canvas
    new_stitched_image[:h1, :w1] = stitched_image

    # Adjust the translation matrix to correctly align the second image
    M[0, 2] = max(0, M[0, 2])
    M[1, 2] = max(0, M[1, 2])

    # Warp the second image using the corrected transformation matrix
    new_image_aligned = cv2.warpAffine(new_image, M, (stitched_w, stitched_h))

    # Create a mask for the new image to handle black areas
    mask_new_image = (new_image_aligned > 0).astype(np.uint8)

    # Combine the aligned new image with the existing stitched image
    new_stitched_image = cv2.bitwise_and(new_stitched_image, new_stitched_image, mask=cv2.bitwise_not(mask_new_image[:, :, 0]))
    new_stitched_image += new_image_aligned

    return new_stitched_image

def capture_and_stitch():
    global stitched_image

    ret, frame = cap.read()
    if ret:
        frame_resized = cv2.resize(frame, (320, 240))
        sharpness = evaluate_sharpness(frame_resized)

        if sharpness >= float(sharpness_threshold.get()):
            if stitched_image is None:
                stitched_image = frame_resized
            else:
                stitched_image = stitch_images(stitched_image, frame_resized)

                # Display stitched image on the canvas
                img = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                canvas.create_image(400, 300, anchor=tk.CENTER, image=imgtk)
                canvas.image = imgtk  # Keep a reference to avoid garbage collection

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
    global cap, stitched_image
    stitched_image = None

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize the GUI
    root = tk.Tk()
    root.title("Microscope Image Stitching")

    global canvas, live_feed_label, sharpness_label, sharpness_threshold, exposure_slider

    # Create a frame to hold the canvas and settings side by side
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Canvas for displaying the images
    canvas = tk.Canvas(main_frame, width=800, height=600)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Frame to hold the settings on the right
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
