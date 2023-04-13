

"""Preprocessing: passes the uploaded image as a {numpy.array}, {PIL.Image} or {str} filepath depending on `type`
-- unless `tool` is `sketch` AND source is one of `upload` or `webcam`.
In these cases, a {dict} with keys `image` and `mask` is passed, and the format of the corresponding values depends on `type`.
"""
import sys
import cv2
import json
import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from pathlib import Path
from torch.utils.data import Dataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model


    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [(Image.fromarray(img.astype(np.uint8))) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(input_image, ddim_steps, num_samples, scale, seed):
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")
    prompt = input_image["prompt"]
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    return result

########################################################
config_path = "./configs/stable-diffusion/v2-inpainting-inference.yaml"
ckpt_path = "./checkpoints/512-inpainting-ema.ckpt"
sampler = initialize_model(config_path, ckpt_path)



json_path = r'./datasets/camo_diff/testjson_dict.json'
size = 512
class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(json_path, 'rt') as f:
            for line in f:
                self.data = json.loads(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        mask_filename = item['target'].replace("target", "mask").replace("jpg", "png")
        prompt = item['prompt']

        #source = cv2.imread(os.path.join(r"./datasets", source_filename.replace("\\", "/")))
        #target = cv2.imread(os.path.join(r"./datasets", target_filename.replace("\\", "/")))
        #mask = cv2.imread(os.path.join(r"./datasets", mask_filename.replace("\\", "/")), 0)
        target = Image.open(os.path.join(r"./datasets", target_filename.replace("\\", "/")))
        mask = Image.open(os.path.join(r"./datasets", mask_filename.replace("\\", "/")))
        #source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        #target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)



        #source = source.astype(np.float32) / 255.0 #[0, 1].
        #target = (target.astype(np.float32) / 127.5) - 1.0 #[-1, 1].
        #mask = mask.astype(np.float32) / 255.0#[0, 1].
        #mask = np.expand_dims(mask, 2)
        return dict(image=target, mask=mask, prompt=prompt)

dataset = MyDataset()
saved_path = r"/cluster/scratch/denfan/longer_prompt/"
for i in range(len(dataset)):
    input_image = dataset[i]

    #input_image = Image.fromarray(input_image)

    num_samples = 1
    ddim_steps = 50
    scale = 10
    seed = 10000

    inpaint_results = predict(input_image, ddim_steps, num_samples, scale, seed)
    inpaint_results.save(os.path.join(saved_path, prompt))

