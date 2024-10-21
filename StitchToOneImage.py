import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Label, Entry, Button
from PIL import Image, ImageTk

def evaluate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def stitch_images(img1, img2, str):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print(np.shape(gray1))

    # Use SIFT to detect and compute key points and descriptors
    sift = cv2.SIFT_create()
    scale_factor = 0.25
    kp1_full = []
    kp2_full = []
    if str == "r":
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        kp1_full = kp1
        kp2_full = kp2
    else: # scale down highres image
        resize1 = cv2.resize(gray1, (0, 0), fx=scale_factor, fy=scale_factor)
        resize2 = cv2.resize(gray2, (0, 0), fx=scale_factor, fy=scale_factor)
        kp1, des1 = sift.detectAndCompute(resize1, None)
        kp2, des2 = sift.detectAndCompute(resize2, None)
        for kp in kp1:
            kp1_full.append(cv2.KeyPoint(kp.pt[0] / scale_factor, kp.pt[1] / scale_factor, kp.size / scale_factor, kp.angle, kp.response, kp.octave, kp.class_id))
        
        for kp in kp2:
            kp2_full.append(cv2.KeyPoint(kp.pt[0] / scale_factor, kp.pt[1] / scale_factor, kp.size / scale_factor, kp.angle, kp.response, kp.octave, kp.class_id))

    print("Img1 Size: ", np.shape(gray1), "New Size: ", np.shape(gray1), "key points size: ", np.shape(kp1_full), "Descriptors size: ", np.shape(des1))
    print("Img2 Size: ", np.shape(gray2), "key points size: ", np.shape(kp2_full), "Descriptors size: ", np.shape(des2))

    # Find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print("Matches: ", np.shape("matches"))

    # Extract locations of matched keypoints & Compute the affine transformation matrix (translation only)
    src_pts = np.float32([kp1_full[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2_full[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts) #RANSAC is default method

    # Flip the translation
    M[0, 2] = -M[0, 2]  # Flip the translation along the x-axis
    M[1, 2] = -M[1, 2]  # Flip the translation along the y-axis

    # Calculate the size of the stitched image canvas
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Calculate the translation offsets
    translation_x = int(M[0, 2])
    translation_y = int(M[1, 2])

    # Determine canvas size and placement based on translation direction
    if translation_x >= 0 and translation_y >= 0:
        # Moving right and down
        stitched_w = max(w1, w2 + translation_x)
        stitched_h = max(h1, h2 + translation_y)
        x_offset, y_offset = 0, 0

    elif translation_x < 0 and translation_y >= 0:
        # Moving left and down
        stitched_w = max(w1 - translation_x, w2)
        stitched_h = max(h1, h2 + translation_y)
        x_offset, y_offset = abs(translation_x), 0

    elif translation_x >= 0 and translation_y < 0:
        # Moving right and up
        stitched_w = max(w1, w2 + translation_x)
        stitched_h = max(h1 - translation_y, h2)
        x_offset, y_offset = 0, abs(translation_y)

    else:
        # Moving left and up
        stitched_w = max(w1 - translation_x, w2)
        stitched_h = max(h1 - translation_y, h2)
        x_offset, y_offset = abs(translation_x), abs(int(translation_y))

    # Create a canvas large enough to hold both images
    stitched_image = np.zeros((stitched_h, stitched_w, 3), dtype=np.uint8) * 255

    # Place the first image on the canvas
    stitched_image[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1

    # Now directly place the second image based on the translation values
    if translation_x < 0:
        translation_x = 0
    if translation_y < 0:
        translation_y = 0
    stitched_image[translation_y:translation_y+h2, translation_x:translation_x+w2] = img2

    return stitched_image

def capture_and_stitch():
    global first_capture, first_highres

    ret, frame = cap.read()
    if ret:
        originalFrame = frame
        frame_resized = cv2.resize(frame, (320, 240))
        sharpness = evaluate_sharpness(frame_resized)

        if sharpness >= float(sharpness_threshold.get()):
            if first_capture is None:
                first_highres = originalFrame
                first_capture = frame_resized
            else:
                second_highres = originalFrame
                second_capture = frame_resized
                stitched_imageR = stitch_images(first_capture, second_capture, "r")
                cv2.imwrite("firstimgresized.jpg", first_capture)
                cv2.imwrite("scndimgresized.jpg", second_capture)

                cv2.imwrite("stichresize.jpg", stitched_imageR)

                stitched_imageH = stitch_images(first_highres, second_highres, "h")
                cv2.imwrite("firstimghighres.jpg", first_highres)
                cv2.imwrite("scndimghighres.jpg", second_highres)

                cv2.imwrite("stitchhighres.jpg", stitched_imageH)

                # Display stitched image on the canvas
                img = Image.fromarray(cv2.cvtColor(stitched_imageR, cv2.COLOR_BGR2RGB))
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
