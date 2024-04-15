import numpy as np

from transformers import AutoImageProcessor, DPTForSemanticSegmentation, DPTFeatureExtractor, DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from ultralytics import YOLO
from supervision import Detections
from huggingface_hub import hf_hub_download



import torch
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.is_available()



model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model_face = YOLO(model_path)
model_face.to(device)

image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model_face_segmets = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model_face_segmets.to(device)


def face_recognision(img_path: str, output_folder:str):
    image = Image.open(img_path)
    output = model_face(image)
    results = Detections.from_ultralytics(output[0])
    n = 0
    for i in results.xyxy:
        # i = [int(i[0] / 1.15), int(i[1] / 1.25), int(i[2] * 1.15), int(i[3] * 1.1)]
        i = list(map(lambda x: int(x), i))
        image1 = image.crop(i)
        image1.save(f"{output_folder}/face_box/{n}_{img_path.split('/')[-1]}")
        face_recognision_depth(f"{output_folder}/face_box/{n}_{img_path.split('/')[-1]}", output_folder)
        n += 1
        return i[0], i[1], i[2] - i[0], i[3] - i[1]


def face_recognision_depth(img_path: str, output_folder:str):
    image = Image.open(img_path)
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model_face_segmets(**inputs)
    prediction = torch.nn.functional.interpolate(
    outputs.logits,
    size=image.size[::-1],  # Reverse the size of the original image (width, height)
    mode="bicubic",
    align_corners=False
    )
    predicted_classes = torch.argmax(prediction, dim=1) + 1
    predicted_classes = predicted_classes.squeeze().cpu()
    person_mask = (predicted_classes == 2) | (predicted_classes == 12) | (predicted_classes == 3) | \
        (predicted_classes == 4) | (predicted_classes == 5) | (predicted_classes == 6) | \
            (predicted_classes == 7) | (predicted_classes == 8) | (predicted_classes == 9) | \
                (predicted_classes == 10) | (predicted_classes == 11) | (predicted_classes == 13)
    person_mask = person_mask.numpy() * 255
    mask_image = Image.fromarray(person_mask.astype('uint8'))
    mask_image.save(f"{output_folder}/mask/{img_path.split('/')[-1]}")