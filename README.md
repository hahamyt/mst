# MST: Adaptive Multi-Scale Tokens Guided Interactive Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2401.04403-b31b1b.svg)](https://arxiv.org/abs/2401.04403)

Online [Demo](http://img2latex.com/).

## Download weights

### ViT-B-448

[+MST-3s](https://cuhko365-my.sharepoint.com/:u:/g/personal/xulong_cuhk_edu_cn/EfhZlrgHqcpEk3XSTlG70HYBu4hcQZmMPCeIU8nTthab_Q?e=vFeltC)

[+MST+CL-3s](https://cuhko365-my.sharepoint.com/:u:/g/personal/xulong_cuhk_edu_cn/EfdjqnSGsqJIlYXp951M4ecB2Wqy18quvz4Y016d_xrxPw?e=UwPmi7)


## Eval

```shell
python evaluate_model.py NoBRS --gpu=1 --checkpoint=weights/mst-3s+cl-448-cclvs.pth --datasets=DAVIS --cf-n=3 --acf --n-clicks 20 --target-iou 0.9 --inference_size 448
```

## Citation
```bibtex
@article{xu2024mst,
  title={MST: Adaptive Multi-Scale Tokens Guided Interactive Segmentation},
  author={Xu, Long and Li, Shanghong and Chen, Yongquan and Luo, Jun},
  journal={arXiv preprint arXiv:2401.04403},
  year={2024}
}
```
