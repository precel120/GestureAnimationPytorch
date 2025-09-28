import cv2
import glob
from prepare_skeleton import get_landmarks, create_transition_sequence


def create_skeleton_frames():
    source = "./0a9ddce1-2cd4-4b14-9e43-3889b6a6e87c.jpg"
    target = "./0a14f51b-4cf4-4694-adb2-4f3f41ae9f19.jpg"
    source_keypoints = get_landmarks(source)
    target_keypoints = get_landmarks(target)
    img = cv2.imread(source)
    create_transition_sequence(img, source_keypoints, target_keypoints)

def create_animation(frame_folder, output_path, fps=30):
    files = sorted(glob.glob(f"{frame_folder}/*.jpg"))
    frame = cv2.imread(files[0])
    h, w, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in files:
        img = cv2.imread(f)
        out.write(img)
    out.release()