# Train a CycleNet

## Step 1 - Download the dataset

You can download the [CycleFill50k dataset](https://huggingface.co/datasets/sihanxu/fill50k/tree/main), and put it into the following dir:

```
CycleNet/training/cfill50k/prompt.json
CycleNet/training/cfill50k/target/X.png
```

In the folder "fill50k/target", you will have 50k images of filled circles.

![image](https://user-images.githubusercontent.com/103425287/221340033-6efdb02e-712f-495c-a88c-f0046432a0bb.png)

In the "cfill50k/prompt.json", you will have their filenames with their condition prompts and uncondition prompts. 

![image](https://user-images.githubusercontent.com/103425287/221340135-92d88e10-465a-4273-8e0d-7cb856c717db.png)

## Step 2 - Load the dataset

Then you can write a script to load the dataset as following(named "tutorial_dataset.py"):

```python
import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/cfill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_filename = item['image']
        source = item['source']
        target = item['target']

        image = cv2.imread('./training/cfill50k/' + image_filename)

        # Do not forget that OpenCV read images in BGR order.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=image, source=source, txt=target)
```

And you can use the following script to test:

```python
from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[16]
image = item['jpg']
source = item['source']
target = item['txt']
print(image.shape)
print(source)
print(target)
```

The outputs of this simple test on my machine are

```

```

Do not ask us why we use these three names as mentioned in ControlNet - this is related to the dark history of a library called LDM.

## Step 3 - Download the pretrained SD model

Then you can go to the [offical page of Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main), and download ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main).

And you need to use ControlNet to control the net, which can be realized by the script provided by ControlNet like (if your SD filename is "./models/v2-1_512-ema-pruned.ckpt" and you want the script to save the processed model (SD+ControlNet) at location "./models/control_sd21_ini.ckpt":

```
python tool_add_cycle_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/cycle_sd21_ini.ckpt
```

You may also use other filenames as long as the command is "python tool_add_control.py input_path output_path".

The output should be like:

![t5](https://user-images.githubusercontent.com/103425287/221340617-dbbf606d-5c79-4934-a168-4c7aca743fa1.png)

## Step 4 - Train the CycleNet

By using the pytorch lighting, the training is very simple.

You can use the follow code to train the data we built before:

```python
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cycleNet.logger import ImageLogger
from cycleNet.model import create_model, load_state_dict


# Configs
resume_path = './models/cycle_sd21_ini.ckpt'
log_path = './logs'
batch_size_per_gpu = 4
gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = False
only_mid_control = False

if __name__ == "__main__":

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cycle_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size_per_gpu, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq, every_n_train_steps=logger_freq)
    trainer = pl.Trainer(accelerator="gpu", devices=gpus, precision=32, callbacks=[logger], default_root_dir=log_path)
    trainer.fit(model, dataloader)
```

