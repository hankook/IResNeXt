# Anytime Neural Prediction via Slicing Networks Vertically (I-ResNeXt)

[arxiv](https://arxiv.org/abs/1807.02609)

The pioneer deep neural networks (DNNs) have emerged to be deeper or wider for improving their accuracy in various applications of artificial intelligence. However, DNNs are often too heavy to deploy in practice, and it is often required to control their architectures dynamically given computing resource budget, i.e., anytime prediction. While most existing approaches have focused on training multiple shallow sub-networks jointly, we study training thin sub-networks instead. To this end, we first build many inclusive thin sub-networks (of the same depth) under a minor modification of existing multi-branch DNNs, and found that they can significantly outperform the state-of-art dense architecture for anytime prediction. This is remarkable due to their simplicity and effectiveness, but training many thin sub-networks jointly faces a new challenge on training complexity. To address the issue, we also propose a novel DNN architecture by forcing a certain sparsity pattern on multi-branch network parameters, making them train efficiently for the purpose of anytime prediction. In our experiments on the ImageNet dataset, its sub-networks have up to 43.3% smaller sizes (FLOPs) compared to those of the state-of-art anytime model with respect to the same accuracy. Finally, we also propose an alternative task under the proposed architecture using a hierarchical taxonomy, which brings a new angle for anytime prediction. 


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

