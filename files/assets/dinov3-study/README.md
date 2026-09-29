---
license: apache-2.0
library_name: openwam
pipeline_tag: robotics
tags:
  - robotics
  - world-action-model
  - openwam
---

# robotwin_dual_system_joint_self_attention_dinov3_svae

An **OpenWAM-Study** checkpoint from the Q3 visual-encoder comparison, run on RoboTwin 2.0. The encoder is always frozen, so swapping it changes which latent the world stream is asked to predict. The baseline is `robotwin_dual_system_joint_self_attention`.

Encoder: **DINOv3** paired with the optional **S-VAE** compression module.

- Paper: https://arxiv.org/abs/2609.07398
- Code: https://github.com/OpenWAM-Official/OpenWAM
- Project page: https://openwam-official.github.io/

## Citation

```bibtex
@article{wang2026openwam,
  title   = {OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining},
  author  = {Yuran Wang and Siqiao Huang and Mingleyang Li and Chenhao Zhang and Jiaqi Liang and Weiyang Jin and Yue Chen and Xuemin Chi and Donghao Zhou and Qize Yu and Yu-Kai Wang and Yuhan Rui and Shenzhe Yao and Zhen Yuan and Zhenhao Shen and Kefei Zhu and Zijie Zhu and Ning Gao and Xiaowei Chi and Guanqi He and Shanghang Zhang and Hao Dong and Lin Shao and Hang Zhao},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2609.07398}
}
```
