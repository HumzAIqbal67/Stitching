import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Label, Entry, Button
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
        # Case 1: Moving right and down
        stitched_w = max(w1, w2 + int(translation_x))
        stitched_h = max(h1, h2 + int(translation_y))
        x_offset, y_offset = 0, 0

    elif translation_x < 0 and translation_y >= 0:
        # Case 2: Moving left and down
        stitched_w = max(w1 - int(translation_x), w2)
        stitched_h = max(h1, h2 + int(translation_y))
        x_offset, y_offset = abs(int(translation_x)), 0

    elif translation_x >= 0 and translation_y < 0:
        # Case 3: Moving right and up
        stitched_w = max(w1, w2 + int(translation_x))
        stitched_h = max(h1 - int(translation_y), h2)
        x_offset, y_offset = 0, abs(int(translation_y))

    else:
        # Case 4: Moving left and up
        stitched_w = max(w1 - int(translation_x), w2)
        stitched_h = max(h1 - int(translation_y), h2)
        x_offset, y_offset = abs(int(translation_x)), abs(int(translation_y))

    # Create a canvas large enough to hold both images
    stitched_image = np.ones((stitched_h, stitched_w, 3), dtype=np.uint8) * 255

    # Place the first image on the canvas
    stitched_image[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1
    cv2.imwrite("A.jpg", stitched_image)

    # Adjust the translation matrix to correctly align the second image
    M[0, 2] += x_offset
    M[1, 2] += y_offset

    # Warp the second image using the corrected transformation matrix
    img2_aligned = cv2.warpAffine(img2, M, (stitched_image.shape[1], stitched_image.shape[0]))
    cv2.imwrite("B.jpg", img2_aligned)

    # Combine the images by overlaying the aligned second image
    

    img2_aligned_gray = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img2_aligned_gray, dtype=bool)

    if translation_x >= 0 and translation_y >= 0:
        # Case 1: Moving right and down
        for i in range(1, img2_aligned_gray.shape[0] - 1):
            for j in range(1, img2_aligned_gray.shape[1] - 1):
                pixel_value = img2_aligned_gray[i, j]
                
                if pixel_value > 0 and img2_aligned_gray[i - 1, j] > 0 and img2_aligned_gray[i, j - 1] > 0:
                    mask[i, j] = True

    #elif translation_x < 0 and translation_y >= 0:
        # Case 2: Moving left and down

    #elif translation_x >= 0 and translation_y < 0:
        # Case 3: Moving right and up

    else:
        # Case 4: Moving left and up
        mask = (img2_aligned_gray > 0)

    stitched_image[mask] = img2_aligned[mask]
    cv2.imwrite("C.jpg", stitched_image)

    return stitched_image

def capture_and_stitch():
    global first_capture

    ret, frame = cap.read()
    if ret:
        frame_resized = cv2.resize(frame, (320, 240))
        sharpness = evaluate_sharpness(frame_resized)

        if sharpness >= float(sharpness_threshold.get()):
            if first_capture is None:
                first_capture = frame_resized
            else:
                second_capture = frame_resized
                stitched_image = stitch_images(first_capture, second_capture)

                cv2.imwrite("AA.jpg", first_capture)
                cv2.imwrite("BB.jpg", second_capture)

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
    global cap, first_capture
    first_capture = None

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