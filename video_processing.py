import cv2
import os

video_path = '/Users/karim/desktop/eece499/spring-osc-trim.mov'
output_dir = '/Users/karim/desktop/eece499/TCN_SINDy/processed_frames'

cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
        frame_filename = f"{frame_count}.png"
        full_path = os.path.join(output_dir, frame_filename)  
        cv2.imwrite(full_path, gray_frame)  

        cv2.imshow('Grayscale Frame', gray_frame)
        frame_count += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

