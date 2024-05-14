---
tags:
- text-to-video
duplicated_from: cerspense/zeroscope_v2_576w
---

# LLM-grounded Video Diffusion Models
[Long Lian](https://tonylian.com/), [Baifeng Shi](https://bfshi.github.io/), [Adam Yala](https://www.adamyala.org/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Boyi Li](https://sites.google.com/site/boyilics/home) at UC Berkeley/UCSF. **ICLR 2024**.

[Project Page](https://llm-grounded-video-diffusion.github.io/) | [Related Project: LMD](https://llm-grounded-diffusion.github.io/) | [Citation](https://llm-grounded-video-diffusion.github.io/#citation)

This model is based on [zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w) but with additional conditioning from bounding boxes in a [GLIGEN](https://gligen.github.io/) fashion.

Similar to [LLM-grounded Diffusion (LMD)](https://llm-grounded-diffusion.github.io/), LLM-grounded Video Diffusion (LVD)'s boxes-to-video stage allows cross-attention-based bounding box conditioning, which uses Zeroscope off-the-shelf. This huggingface model offers an alternative: we train a GLIGEN model (i.e., transformer adapters) with Zeroscope's weights without the temporal transformers blocks on [SA-1B](https://ai.meta.com/datasets/segment-anything/), treating it as a SD v2.1 model that has been fine-tuned to 256x256 resolution. We then merge the adapters into Zeroscope to offer conditioning. The resulting model is in this hugginface model. This can be used with cross-attention-based conditioning or on its own, similar to [LMD+](https://github.com/TonyLianLong/LLM-groundedDiffusion). This can be used with LLM-based text-to-dynamic scene layout generator in LVD, or on its own as a video version of GLIGEN.

## Citation (LVD)
If you use our work, model, or our implementation in this repo, or find them helpful, please consider giving a citation.
```
@article{lian2023llmgroundedvideo,
      title={LLM-grounded Video Diffusion Models}, 
      author={Lian, Long and Shi, Baifeng and Yala, Adam and Darrell, Trevor and Li, Boyi},
      journal={arXiv preprint arXiv:2309.17444},
      year={2023},
}

@article{lian2023llmgrounded,
    title={LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models}, 
    author={Lian, Long and Li, Boyi and Yala, Adam and Darrell, Trevor},
    journal={arXiv preprint arXiv:2305.13655},
    year={2023}
}
```

## Citation (GLIGEN)
The adapters in this model are trained in a mannar similar to training GLIGEN adapters.
```
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}
```

## Citation (ModelScope)
ModelScope is LVD's base model.

```
@article{wang2023modelscope,
    title={Modelscope text-to-video technical report},
    author={Wang, Jiuniu and Yuan, Hangjie and Chen, Dayou and Zhang, Yingya and Wang, Xiang and Zhang, Shiwei},
    journal={arXiv preprint arXiv:2308.06571},
    year={2023}
}
@InProceedings{VideoFusion,
    author    = {Luo, Zhengxiong and Chen, Dayou and Zhang, Yingya and Huang, Yan and Wang, Liang and Shen, Yujun and Zhao, Deli and Zhou, Jingren and Tan, Tieniu},
    title     = {VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023}
}
```

## LICENSE
Zeroscope follows CC-BY-NC 4.0 license. The gligen adapters are trained on SA-1B, which follows [SA-1B license](https://ai.meta.com/datasets/segment-anything/).
