# CycleNet

CycleNet: Cycle-Consistent Diffusion for Unpaired Image-to-Image Translation

- This repository is still under construction.

## [Website](https://cyclenetweb.github.io) | [Paper]()

## Conda Environment

```
conda env create -f environment.yaml
conda activate cycle
```

## How to train

[Train the model](./docs/train.md)

## References

```
@inproceedings{cyclenet,
    title = "CycleNet: Rethinking Cycle Consistent in Text‑Guided Diffusion for Image Manipulation",
    author = "Xu, Sihan  and
      Ma, Ziqiao  and
      Huang, Yidong  and
      Lee, Honglak  and
      Chai, Joyce",
    booktitle = "Advances in Neural Information Processing Systems",
    year = "2023",
}
```

This implementation is based on [ControlNet](https://github.com/lllyasviel/ControlNet).
