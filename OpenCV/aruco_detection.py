import cv2
import numpy as np

# ─────────────────────────────
# All ArUco dictionaries
# ─────────────────────────────
ARUCO_DICTS = {
    "4x4": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    "5x5": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    "6x6": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
}

parameters = cv2.aruco.DetectorParameters()

# Marker size in meters
MARKER_SIZE = 0.05  # 5cm

# ─────────────────────────────
# Open phone camera
# ─────────────────────────────
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get actual frame size
ret, frame = cap.read()
h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# ─────────────────────────────
# Camera matrix
# ─────────────────────────────
camera_matrix = np.array([
    [w,  0,  w/2],
    [0,  w,  h/2],
    [0,  0,    1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))

# ─────────────────────────────
# Main loop
# ─────────────────────────────
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect across ALL dictionaries ──
    all_corners = []
    all_ids = []
    all_types = []

    for dict_name, aruco_dict in ARUCO_DICTS.items():
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for j, corner in enumerate(corners):
                all_corners.append(corner)
                all_ids.append(ids[j][0])
                all_types.append(dict_name)

    # ── Draw and annotate each detected marker ──
    if len(all_ids) > 0:
        for i, corner in enumerate(all_corners):

            # Draw green border
            cv2.aruco.drawDetectedMarkers(frame, [corner],
                                          np.array([[all_ids[i]]]))

            # Pose estimation
            obj_points = np.array([
                [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
            ], dtype=np.float32)

            img_points = corner[0].astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Draw 3D axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec, 0.005)

                # Distance in cm
                distance = np.linalg.norm(tvec) * 100

                # Rotation angles
                rot_mat, _ = cv2.Rodrigues(rvec)
                sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
                rx = np.degrees(np.arctan2(rot_mat[2,1], rot_mat[2,2]))
                ry = np.degrees(np.arctan2(-rot_mat[2,0], sy))
                rz = np.degrees(np.arctan2(rot_mat[1,0], rot_mat[0,0]))

                # Marker center
                c = corner[0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))

                # Show tag type (4x4 / 5x5 / 6x6)
                cv2.putText(frame,
                            f"{all_types[i]} ID:{all_ids[i]}",
                            (cx - 60, cy - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 100, 0), 2)

                # Show distance
                cv2.putText(frame,
                            f"Dist: {distance:.1f} cm",
                            (cx - 60, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                # Show rotation
                cv2.putText(frame,
                            f"Rx:{rx:.1f} Ry:{ry:.1f} Rz:{rz:.1f}",
                            (cx - 60, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1)

                print(f"[{all_types[i]}] ID:{all_ids[i]} "
                      f"Dist:{distance:.1f}cm "
                      f"Rx:{rx:.1f} Ry:{ry:.1f} Rz:{rz:.1f}")
    else:
        cv2.putText(frame, "No marker found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("ArUco Pose Estimation - 4x4 | 5x5 | 6x6", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("aruco_screenshot.png", frame)
        print("Screenshot saved!")

cap.release()
cv2.destroyAllWindows()