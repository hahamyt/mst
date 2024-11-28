# MST: Adaptive Multi-Scale Tokens Guided Interactive Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2401.04403-b31b1b.svg)](https://arxiv.org/abs/2401.04403)

Online [Demo](http://img2latex.com/).

## Download weights

### ViT-B-1024

[+MST-3s](https://cuhko365-my.sharepoint.com/:u:/g/personal/xulong_cuhk_edu_cn/EQPswrQ8MsJHhdZNKIgdBIoB5lGLpwIcrWXyhXmNjPQbpg?e=aQ1RtW)

[+MST+CL-3s](https://cuhko365-my.sharepoint.com/:u:/g/personal/xulong_cuhk_edu_cn/EbyTj7JJlmxMl36e2cRNTogBLd7D_A0M51AxBjsmqP7Giw?e=t2p63a)


## Eval

```shell
python evaluate_model.py NoBRS --gpu=1 --checkpoint=weights/mst+cl-B-3s-1024.pth --datasets=DAVIS --cf-n=3 --acf --n-clicks 20 --target-iou 0.9 --inference_size 1024
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
