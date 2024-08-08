import cv2
import time

def display_video_from_camera(camera_index=0, width=640, height=480):
    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    # Set desired frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set desired frame rate (this is a request; actual rate may vary)
    desired_fps = 30
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # Initialize variables to calculate frame rate
    frame_count = 0
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Resize the frame (optional)
        frame = cv2.resize(frame, (width, height))

        # Increment frame count
        frame_count += 1

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate frame rate
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        # Display the frame rate on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Camera Feed', cv2.WND_PROP_VISIBLE) < 1:
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera_index = int(input("Enter the camera index: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
    else:
        # You can adjust the width and height as needed
        display_video_from_camera(camera_index, width=320, height=240)
