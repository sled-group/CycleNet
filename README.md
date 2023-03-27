# CycleNet
Official repo of CycleNet 

- CycleNet: trainable model for image-to-image translation with unpaired dataset

## Conda Environment

```
conda env create -f environment.yaml
conda activate cycle
```

## Models

For Pix2Pix task:
- Translation: cycle_v21_pix_t.yaml
- Object Edit: cycle_v21_pix_e.yaml
- Style Transfer: cycle_v21_pix_c.yaml

For Non Pix2Pix task:
- Translation: cycle_v21_nonpix.yaml

## How to train

[Train the model](./docs/train.md)
