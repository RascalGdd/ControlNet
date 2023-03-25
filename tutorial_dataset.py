import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset


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


        source = cv2.imread(os.path.join(r"./datasets", source_filename.replace("\\", "/")))
        target = cv2.imread(os.path.join(r"./datasets", target_filename.replace("\\", "/")))
        mask = cv2.imread(os.path.join(r"./datasets", mask_filename.replace("\\", "/")), 0)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # resize the image
        source = cv2.resize(source, (size, size))
        target = cv2.resize(target, (size, size))
        mask = cv2.resize(mask, (size, size))


        # Normalize source images to [0, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0

        # Normalize target images to [0, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # Normalize mask images to [0, 1].
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, 2)

        return dict(jpg=target, txt=prompt, hint=source, mask=mask)

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
mask = item['mask']
print(txt)
print(jpg.shape)
print(hint.shape)
print(mask.shape)