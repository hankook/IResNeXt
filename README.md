# Anytime Neural Prediction via Slicing Networks Vertically (I-ResNeXt)

[arxiv](https://arxiv.org/abs/1807.02609)

## Requirements

```bash
conda create -n iresnext python=3.6
conda activate iresnext
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
pip install tensorboard
```

**Note.** The reported results in [paper](https://arxiv.org/abs/1807.02609) were obtained in 2018 with an old-version of Pytorch.

## Experiments (I-ResNeXt)

The results reported in Section 4.2 can be obtained using following commands:

```sh
python main2_anytime.py --model AnytimeResNeXt --dataset cifar10 --logdir logs/cifar10_AnytimeRexNeXt_L21 --num_blocks 7
python main2_anytime.py --model AnytimeResNeXt --dataset cifar100 --logdir logs/cifar100_AnytimeRexNeXt_L21 --num_classes 100 --num_blocks 7
```

