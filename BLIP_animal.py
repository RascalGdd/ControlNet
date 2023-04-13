import json
import torch
import os
from lavis.models import load_model_and_preprocess
from PIL import Image, ImageChops, ImageOps
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = r"/cluster/work/cvl/denfan/diandian/ControlNet/datasets/camo_diff/"
mask_path = os.path.join(folder_path, "camo_mask")
target_path = os.path.join(folder_path, "camo_target")
source_path = os.path.join(folder_path, "camo_source")
target_mask_path = os.path.join(folder_path, "target_mask")


def separating_fore(mask, raw_img, imgname):
    # foreground=Image.composite(raw_img, raw_img, mask)
    foreground = Image.composite(raw_img, Image.new('RGB', raw_img.size, 'black'), mask)
    foreground.save(os.path.join(target_mask_path, imgname))
    return foreground


def caption_word(raw_img):
    # BLIP caption
    # loads BLIP caption base model, with finetuned ckpt and image processors
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                         device=device)
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
    # generate caption
    prompt = model.generate({"image": image})  # ['a large fountain spewing water into the air']
    return prompt[0]


def color_word(raw_img):
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True,
                                                                      device=device)
    question = "The animal color is {}"
    image = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    prompt = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    return prompt[0]


# json
index=0
file_list = []
source_folder = 'camo_diff\\camo_source\\'
target_folder = 'camo_diff\\camo_target\\'
for maskname in os.listdir(mask_path):
    imgname = maskname.replace("png", "jpg")
    target_image = Image.open(os.path.join(target_path, imgname))
    mask_image = Image.open(os.path.join(mask_path, maskname))

    # given name
    parts = maskname.split('-')
    text = ' '.join([parts[3], parts[5]])

    foreground = separating_fore(mask=mask_image, raw_img=target_image, imgname=imgname)
    # caption
    caption = caption_word(raw_img=target_image)
    
    #model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True,
                                                                      #device=device)
    #question = "describe the animal {}"
    #image = vis_processors["eval"](foreground).unsqueeze(0).to(device)
    #question = txt_processors["eval"](question)
    #caption = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    #caption =caption[0]
    # color
    animal_color = color_word(raw_img=foreground)
    #prompt = caption + ", the camoflage " + text + " is " + animal_color
    prompt="the camoflage " + text + " is " + animal_color+", and looks like "+caption
    file_list.append({"source": source_folder + imgname, "target": source_folder + imgname, "prompt": prompt})
    print(prompt)
    index+=1
    if index==5:
        break
with open('./datasets/camo_diff/' + "animal_caption.json", "a") as f:
    json.dump(file_list, f)

# for maskname in os.listdir(mask_path):
#     imgname = maskname.replace("png", "jpg")
#     target_image = Image.open(os.path.join(target_path, imgname))
#     mask_image = Image.open(os.path.join(mask_path, maskname))
#
#     "given name"
#     parts = maskname.split('-')
#     text = ' '.join([parts[3], parts[5]])
#     "caption"
#     caption = caption_word(raw_img=target_image)
#     "color"
#     foreground = separating_fore(mask=mask_image, raw_img=target_image, imgname=imgname)
#     animal_color = color_word(raw_img=foreground)
#     prompt_fore = caption + ", the camoflage " + text + " is " + animal_color
#     file_list_fore.append(
#         {"source": os.path.join(source_folder, imgname), "target": os.path.join(source_folder, imgname),
#          "prompt": prompt_fore})
#     print(prompt_fore)
# with open('./datasets/camo_diff/' + "caption_fore.json", "a") as f:
#     json.dump(file_list_fore, f)




