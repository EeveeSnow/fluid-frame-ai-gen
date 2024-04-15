import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# from transformers import DPTFeatureExtractor, DPTForSemanticSegmentation
# from transformers import AutoImageProcessor, DPTForSemanticSegmentation, DPTFeatureExtractor, DetrImageProcessor, DetrForObjectDetection
# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# from ultralytics import YOLO
# from supervision import Detections
# from huggingface_hub import hf_hub_download

# from torchvision.transforms import ToTensor, ToPILImage
import os
# import torch
from PIL import Image

from face import face_recognision
from person import person_recognision
from grid import *




def create_folders(folder_name):
    subfolders = ['ai', 'comb', 'face_box', 'grid', 'grid_ai', 'grid_mask', 'mask', 'cropped']
    for subfolder in subfolders:
        os.makedirs(os.path.join(folder_name, subfolder), exist_ok=True)
        

def load_video_face(filename:str, output_folder:str):
    video = cv2.VideoCapture(filename)
    frame_count = 0
    video_d = open('data.txt', "w")
    line = ""
    create_folders(output_folder)
    while True:
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(f"{output_folder}/frame{frame_count}.jpg", frame)
        box = face_recognision(f"{output_folder}/frame{frame_count}.jpg", output_folder)
        line += str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n"
        frame_count += 1
    video_d.write(line)
    video_d.close()
    video.release()


def load_video_person(filename:str, output_folder:str):
    video = cv2.VideoCapture(filename)
    frame_count = 0
    video_d = open('data.txt', "w")
    line = ""
    create_folders(output_folder)
    while True:
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(f"{output_folder}/frame{frame_count}.jpg", frame)
        box = person_recognision(f"{output_folder}/frame{frame_count}.jpg", output_folder)
        line += str(box[0]) + " " + str(box[1])  + " " + str(box[2]) + " " + str(box[3]) + "\n"
        frame_count += 1
    video_d.write(line)
    video_d.close()
    video.release()


def make_video(filename:str, img_folder:str, frames: int, res: tuple, fps:int):
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, res)
    video_d = open('data.txt', "r")
    boxes = list(map(lambda x: list(map(lambda x: int(x), x.split())), video_d.readlines()))
    for i in range(frames):
        back = Image.open(f"{img_folder}/frame{i}.jpg")
        if i >= 10 and i < 100:
            ai = f"000{i}-frame{i}"
        elif i >= 100:
            ai = f"00{i}-frame{i}"
        elif i < 10:
            ai = f"0000{i}-frame{i}" 
        ai = Image.open(f"{img_folder}/ai/{ai}.png")
        back.paste(ai, boxes[i][0:2])
        back.save(f"{img_folder}/comb/frame{i}.jpg")
        video.write(cv2.imread(f"{img_folder}/comb/frame{i}.jpg"))
    cv2.destroyAllWindows()
    video.release()


def make_video_grid(filename:str, img_folder:str, frames: int, res: tuple, fps:int, n: int):
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, res)
    video_d = open('data.txt', "r")
    boxes = list(map(lambda x: list(map(lambda x: int(x), x.split())), video_d.readlines()))
    for i in range(frames):
        back = Image.open(f"{img_folder}/frame{i}.jpg")
        ai = Image.open(f"{img_folder}/ai/frame{i}.png")
        back.paste(ai, boxes[(n+1)*i][0:2])
        back.save(f"{img_folder}/comb/frame{i}.jpg")
        video.write(cv2.imread(f"{img_folder}/comb/frame{i}.jpg"))
    cv2.destroyAllWindows()
    video.release()


