import torch
import numpy as np
import os
import PIL.Image as Image
import cv2
from diffusers import StableDiffusionInpaintPipeline
import timm.models
import torch.nn as nn
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# raw_image = Image.open(r"C:\Users\guodi\Desktop\camouflaged_dataset\camo_diff\camo_source\COD10K-CAM-1-Aquatic-1-BatFish-3.jpg").convert("RGB")
# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

# question = "The background color is {}"
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# question = txt_processors["eval"](question)
# color_word = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
#
with torch.autocast("cuda"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
    )
# prompt = "a " + color_word[0] + " fish"
prompt = "uniform texture, uniform color"
folder_path = r"/cluster/work/cvl/denfan/diandian/ControlNet/datasets/camo_diff/"
mask_path = os.path.join(folder_path, "camo_mask")
img_path = os.path.join(folder_path, "camo_target")
for maskname in os.listdir(mask_path):
    imgname = maskname.replace("png", "jpg")

    image = Image.open(os.path.join(img_path, imgname)).resize([512, 512])
    mask_image = Image.open(os.path.join(mask_path, maskname)).resize([512, 512])
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    image.save("/cluster/scratch/denfan/diandian_output/" + imgname)
