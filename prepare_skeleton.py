import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

def interpolate_skeletons(points_A, points_B, num_frames=30):
    """
    Interpolates between two skeletons.

    Args:
        points_A (np.ndarray): (21, 2) array of keypoints for gesture A
        points_B (np.ndarray): (21, 2) array of keypoints for gesture B
        num_frames (int): number of frames in the sequence

    Returns:
        list of np.ndarray: sequence of interpolated keypoints
    """
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        points_t = (1 - t) * points_A + t * points_B
        frames.append(points_t.astype(int))
    return frames

def points_to_landmark_list(points, image_size):
    """
    Convert interpolated (x,y) pixel coords into a NormalizedLandmarkList
    for mp_drawing.draw_landmarks.
    """
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for (x, y) in points:
        landmark = landmark_pb2.NormalizedLandmark()
        landmark.x = x / image_size
        landmark.y = y / image_size
        landmark.z = 0  # we don’t interpolate depth here
        landmark_list.landmark.append(landmark)
    return landmark_list

def remove_hand_and_draw_interpolated_skeleton(image_bgr, points, expand=60, fill_color=(255,255,255)):
    h, w = image_bgr.shape[:2]

    # --- STEP 1: mask ---
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- STEP 2: remove hand ---
    out = image_bgr.copy()
    out[mask == 255] = fill_color

    # --- STEP 3: draw skeleton using MediaPipe style ---
    points = points_to_landmark_list(points, h)
    mp_drawing.draw_landmarks(out, points, mp_hands.HAND_CONNECTIONS)

    return out

def remove_hand_and_draw_skeleton_blank(path, it, expand=60, fill_color=(255,255,255)):
    if os.path.exists(path) == False:
        return
    image_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks or it >= len(result.multi_hand_landmarks):
        return
    
    return remove_hand_and_draw_interpolated_skeleton(image_bgr, result.multi_hand_landmarks[it], expand=expand, fill_color=fill_color)

def replace_gesture(source_gesture, target_gesture, expand=60, fill_color=(255,255,255)):
    image_bgr = cv2.imread(target_gesture)
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    # --- STEP 1: create mask ---
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in result.multi_hand_landmarks:
        pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    # expand mask so it covers entire hand
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- STEP 2: remove hand by filling ---
    out = cv2.imread(source_gesture)
    out[mask == 255] = fill_color  # paint hand area white (or black)

    # --- STEP 3: draw skeleton ---
    for hand in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(out, hand, mp_hands.HAND_CONNECTIONS)

    return out

# replace_gesture('./0a9ddce1-2cd4-4b14-9e43-3889b6a6e87c.jpg', './0a14f51b-4cf4-4694-adb2-4f3f41ae9f19.jpg', 40)

def create_transition_sequence(img, source_keypoints, target_keypoints,  out_dir="./gesture_seq", num_frames=30, expand=60):
    # Interpolate
    seq = interpolate_skeletons(source_keypoints, target_keypoints, num_frames=num_frames)
    os.makedirs(out_dir, exist_ok=True)

    # Generate frames
    for i, pts in enumerate(seq):
        frame = remove_hand_and_draw_interpolated_skeleton(img, pts, expand=expand)
        save_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(save_path, frame)

    print(f"Saved {num_frames} masked skeleton frames to {out_dir}")

def get_landmarks(path):
    if os.path.exists(path) == False:
        return
    image_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return
    
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in result.multi_hand_landmarks[0].landmark], dtype=np.int32)