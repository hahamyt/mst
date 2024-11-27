# MST: Adaptive Multi-Scale Tokens Guided Interactive Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2401.04403-b31b1b.svg)](https://arxiv.org/abs/2401.04403)

Codes and data will be coming in few days...


Online [Demo](http://img2latex.com/).


```shell
python evaluate_model.py NoBRS --gpu=1 --checkpoint=weights/upconv-3s.pth --datasets=DAVIS --cf-n=3 --acf --n-clicks 20 --target-iou 0.9 --inference_size 224
```

[3s (intervel = 4), 6s (intervel = 2)](isegm/model/modeling/models.py)

```shell
python evaluate_model.py NoBRS --gpu=1 --checkpoint=weights/upconv-6s.pth --datasets=DAVIS --cf-n=3 --acf --n-clicks 20 --target-iou 0.9 --inference_size 224
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
