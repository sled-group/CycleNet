# CycleNet
**This repository is still actively under construction!**

This is the official implementation of:

CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation  
Sihan Xu*, Ziqiao Ma*, Yidong Huang, Honglak Lee, Joyce Chai  
University of Michigan, LG AI Research  
[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/69913)

### [Project Page](https://cyclenetweb.github.io) | [Paper](http://arxiv.org/abs/2310.13165) | [ManiCups Dataset](https://huggingface.co/datasets/sled-umich/ManiCups) | [Models](https://huggingface.co/sled-umich/CycleNet)

## Conda Environment

```
conda env create -f environment.yaml
conda activate cycle
```

## Training

This implementation builds upon [ControlNet](https://github.com/lllyasviel/ControlNet).  
Please redirect to this document on how to [train the model](./docs/train.md).

## Citation

```
@inproceedings{xu2023cyclenet,
    title = "CycleNet: Rethinking Cycle Consistent in Text‑Guided Diffusion for Image Manipulation",
    author = "Xu, Sihan and Ma, Ziqiao and Huang, Yidong and Lee, Honglak and Chai, Joyce",
    booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
    year = "2023",
}
```
