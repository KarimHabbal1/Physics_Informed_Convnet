import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = '/Users/karim/desktop/eece499/spring-osc-trim.mov'


# === Step 1: Select Reference Point ===
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to load first frame.")

ref_point = []

def click_reference(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.clear()
        ref_point.append((x, y))
        print(f"Reference point selected at: ({x}, {y})")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Reference", frame)

cv2.imshow("Select Reference", frame)
cv2.setMouseCallback("Select Reference", click_reference)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not ref_point:
    raise ValueError("No reference point selected.")

ref_x = ref_point[0][0]
ref_y = ref_point[0][1]

# === Step 2: Select Rod Template with Click + Drag ===
roi = cv2.selectROI("Select Rod Template", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = map(int, roi)
template = frame[y:y+h, x:x+w]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# === Step 3: Template Match Across All Frames ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps

positions = []
times = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2

    displacement = center_x - ref_x
    positions.append(displacement)
    times.append(frame_idx * dt)

    # 🔴 Draw red dot (tracked rod center)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"x: {displacement}px", (center_x + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 🟢 Draw green dot (reference point)
    cv2.circle(frame, (ref_x, ref_y), 5, (0, 255, 0), -1)
    cv2.putText(frame, "Ref", (ref_x + 10, ref_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 🖼️ Show tracking
    cv2.imshow("Rod Tracking - Template Match", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# === Step 4: Plot Displacement and Velocity ===
positions = np.array(positions)
times = np.array(times)
velocities = np.gradient(positions, dt)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(times, positions)
plt.title("x(t): Rod Displacement")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (pixels)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, velocities, color='orange')
plt.title("v(t): Rod Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (pixels/s)")
plt.grid(True)

plt.tight_layout()
plt.show()