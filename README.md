# Towards Real-time Video Compressive Sensing on Mobile Devices (ACM MM 2024)
Miao Cao, Lishun Wang, Huan Wang, Guoqing Wang and Xin Yuan

<hr />

> **Abstract:** Video Snapshot Compressive Imaging (SCI) uses a low-speed 2D camera to capture high-speed scenes as snapshot compressed measurements, followed by a reconstruction algorithm to retrieve the high-speed video frames. The fast evolving mobile devices and existing high-performance video SCI reconstruction algorithms
motivate us to develop mobile reconstruction methods for real-world applications. Yet, it is still challenging to deploy previous reconstruction algorithms on mobile devices due to the complex inference process, let alone real-time mobile reconstruction. To the best of our knowledge, there is no video SCI reconstruction model designed to run on the mobile devices. Towards this end, in this paper, we present an effective approach for video SCI reconstruction, dubbed MobileSCI, which can run at real-time speed on the mobile devices for the first time. Specifically, we first build a U-shaped 2D convolution-based architecture, which is much more efficient and mobile-friendly than previous state-of-the-art reconstruction methods. Besides, an efficient feature mixing block, based on the channel splitting and shuffling mechanisms, is introduced as
a novel bottleneck block of our proposed MobileSCI to alleviate the computational burden. Finally, a customized knowledge distillation strategy is utilized to further improve the reconstruction quality. Extensive results on both simulated and real data show that our proposed MobileSCI can achieve superior reconstruction quality with high efficiency on the mobile devices. Particularly, we can reconstruct a 256×256×8 snapshot compressed measurement with real-time performance (about 35 FPS) on an iPhone 15.
<hr />

## Installation
Please see the [Installation Manual](docs/install.md) for MobileSCI Installation. 

## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/\_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/mobile_sci/mobile_sci.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/mobile_sci/mobile_sci.py 
```

## Testing MobileSCI on Grayscale Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in grayscale simulation dataset by executing the statement below.

```
python tools/test.py configs/mobile_sci/mobile_sci.py --weights=checkpoints/mobilesci_base.pth
```

## Citation

```
@inproceedings{cao2024towards,
  title={Towards Real-time Video Compressive Sensing on Mobile Devices},
  author={Cao, Miao and Wang, Lishun and Wang, Huan and Wang, Guoqing and Yuan, Xin},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
```
