from dataclasses import dataclass
from typing import List, Dict

IMAGES_PATH = './images'
CROPPED_PATH = './cropped'
LANDMARKS_PATH = './landmarks'
ANNOTATIONS_PATH = './annotations'
CHECKPOINT_PATH = './checkpoints'

GESTURES = ['call', 'three_gun']

@dataclass
class Record:
    bboxes: List[List[float]]  # List of [x, y, width, height]
    labels: List[str]
    hand_landmarks: List[List[List[float]]]  # 2D landmarks per hand
DataStructure = Dict[str, Record]