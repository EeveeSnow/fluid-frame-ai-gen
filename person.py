import numpy as np

# from transformers import DPTFeatureExtractor, DPTForSemanticSegmentation
from transformers import AutoImageProcessor, DPTForSemanticSegmentation, DPTFeatureExtractor, DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation



import torch
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.is_available()

processor = AutoImageProcessor.from_pretrained("Intel/dpt-large-ade")
model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
model.to(device)

processor_f = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model_f = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model_f.to(device)

def person_recognision(img_path: str, output_folder:str):
    image = Image.open(img_path)
    inputs = processor_f(images=image, return_tensors="pt").to(device)
    outputs = model_f(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor_f.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        if "person" in model_f.config.id2label[label.item()]:
            image1 = image.crop(box)
            image1.save(f"{output_folder}/cropped/{img_path.split('/')[-1]}")
            mask_image = person_recognision_depth(f"{output_folder}/cropped/{img_path.split('/')[-1]}", output_folder)
            mask_image.save(f"{output_folder}/mask/{img_path.split('/')[-1]}")
            return box[0], box[1], box[2] - box[0], box[3] - box[1]


def person_recognision_depth(img_path: str, output_folder:str):
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    prediction = torch.nn.functional.interpolate(
    outputs.logits,
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False
    )
    predicted_classes = torch.argmax(prediction, dim=1) + 1
    predicted_classes = predicted_classes.squeeze().cpu()
    person_mask = predicted_classes == 13   
    person_mask = person_mask.numpy() * 255
    mask_image = Image.fromarray(person_mask.astype('uint8'))
    return mask_image