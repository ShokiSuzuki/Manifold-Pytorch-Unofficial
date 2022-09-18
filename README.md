# Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation

This is an unofficial PyTorch implementation of "[Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation](https://arxiv.org/abs/2107.01378)".


## Usage

### Train

```
./sample.sh
```

### Evaluate

```
python train.py --eval --models deit_tiny_patch16_224 --resume <checkpoint path> --data-path <dataset path>
```


## Result


|   | Teacher Model | Top-1 (%) | Student Model | Top-1 (%) |
|:---:|:---:|:---:|:---:|:---:|
| Paper | CaiT-XXS24 | 78.5 | DeiT-Tiny | 75.5 |
| This Code | CaiT-XXS24 | 78.5 | DeiT-Tiny | 75.1 ([link](https://drive.google.com/file/d/142QnSh6sIxQxKiazS7Dv4VmWU9S5hzix/view?usp=sharing)) |
