import cv2
import numpy as np
import matplotlib.pyplot as plt

def variance_of_laplacian(image):
    """Compute the Laplacian of the image and return the variance."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def capture_frames(video_path, start_frame, end_frame, step):
    """Capture frames from the video at specified intervals."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame, step):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((i, frame, gray))
    cap.release()
    return frames

def analyze_focus(frames):
    """Analyze the focus of each frame and return a list of focus metrics."""
    focus_measures = []
    for idx, color_frame, gray_frame in frames:
        focus_measure = variance_of_laplacian(gray_frame)
        focus_measures.append((idx, color_frame, focus_measure))
    return focus_measures

def plot_focus_images(focus_measures):
    """Plot images with their focus metrics."""
    num_images = len(focus_measures)
    fig, axes = plt.subplots(4, 5, figsize=(20, 8))
    axes = axes.flatten()

    for ax, (idx, image, focus) in zip(axes, focus_measures):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {idx}\nFocus: {focus:.2f}", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    video_path = 'C:\\Users\\humza\\OneDrive\\Desktop\\Auto-Pumping-SickKids\\IMG_1898.mov'
    start_frame = 200
    end_frame = start_frame + 600
    step = 20

    # Capture frames
    frames = capture_frames(video_path, start_frame, end_frame, step)

    # Analyze focus
    focus_measures = analyze_focus(frames)

    # Plot the images with their focus metrics
    plot_focus_images(focus_measures)

if __name__ == "__main__":
    main()
