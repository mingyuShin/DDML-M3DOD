# Depth-discriminative Metric Learning for Monocular 3D Object Detection

By [Wonhyeok Choi](https://wonhyeok-choi.github.io/), [Mingyu Shin](https://mingyushin.github.io/), [Sunghoon Im](https://sunghoonim.github.io/).


## Introduction

This repository is an official implementation of the paper ['Depth-discriminative Metric Learning
for Monocular 3D Object Detection'](https://openreview.net/forum?id=ZNBblMEP16) using ['Delving into Localization Errors for Monocular 3D Detection'](https://arxiv.org/abs/2103.16237). In this work, we address the challenge of monocular 3D object detection in RGB images by proposing a novel metric learning scheme. Our method, which does not rely on extra parameters, modules, or data, concentrates on extracting depth-discriminative features without increasing the inference time or model size.

<img src="resources/example.png" alt="vis" style="zoom:50%;" />




## Usage

### Installation
This repo is tested on our local environment (python=3.6, cuda=9.2, pytorch=1.10.0), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n qi_monodle python=3.6
```
Then, activate the environment:
```bash
conda activate qi_monodle
```

Install  Install PyTorch:

```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |training/
        |calib/
        |image_2/
        |label/
      |testing/
        |calib/
        |image_2/
```

Make the object-wise depth map following command:
```sh
cd #ROOT
python make_obj_depth.py
```

### Training & Evaluation

Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 python ../../tools/train_val.py --config kitti_example.yaml
```
<!-- The model will be evaluated automatically if the training completed. If you only want evaluate your trained model (or the provided [pretrained model](https://drive.google.com/file/d/1jaGdvu_XFn5woX0eJ5I2R6wIcBLVMJV6/view?usp=sharing)) , you can modify the test part configuration in the .yaml file and use the following command: -->

The model will be evaluated automatically if the training completed. If you only want evaluate your trained model, you can modify the test part configuration in the .yaml file and use the following command:

```sh
python ../../tools/train_val.py --config kitti_example.yaml --e
```

For ease of use, we also provide a pre-trained checkpoint, which can be used for evaluation directly. See the below table to check the performance.

|                   | AP40@Easy | AP40@Mod. | AP40@Hard |
| ----------------- | --------- | --------- | --------- |
| Monodle           | 17.32     | 14.35     | 12.22     |
| Monodle + Ours    | 21.31     | 16.53     | 13.93     |

## Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{choi2023depth,
title={Depth-discriminative Metric Learning for Monocular 3D Object Detection},
author={Choi, Wonhyeok and Shin, Mingyu and Im, Sunghoon},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023}
}
```

## Acknowlegment

This repo benefits from the excellent work [Monodle](https://github.com/xinzhuma/monodle). Please also consider citing it.

## License

This project is released under the MIT License.

## Contact

If you have any question about this project, please feel free to contact smu06117@dgist.ac.kr or alsrb4446@dgist.ac.kr.
