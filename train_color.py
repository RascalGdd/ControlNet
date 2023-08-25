from share import *
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_color import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = "/cluster/work/cvl/denfan/diandian/control/T2I-COD/models/sd-v1-5-inpainting.ckpt"
#resume_path = "./models/control_v11p_sd15_inpaint.pth"
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = True


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_9.yaml').cpu()
#model = create_model('./models/control_v11p_sd15_inpaint.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=4, strategy="ddp", precision=32, callbacks=[logger], resume_from_checkpoint=resume_path)


# Train!
trainer.fit(model, dataloader)
