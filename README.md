# Extentions on: [MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection](https://github.com/ZrrSkywalker/MonoDETR)
Project for the Advanced Deep Learning for Computer Vision course of DTU.
* Rolando Esquivel
* Esteban Zamora
* Lucas Sandby
* Jonathan Mikler

## Introduction (From the original repo)
MonoDETR is the **first DETR-based model** for monocular 3D detection **without additional depth supervision, anchors or NMS**. We enable the vanilla transformer in DETR to be depth-guided and achieve scene-level geometric perception. In this way, each object estimates its 3D attributes adaptively from the depth-informative regions on the image, not limited by center-around features.
<div align="center">
  <img src="main_fig.png"/>
</div>


## Installation
1. Clone this project and create a conda environment:
    ```
    git clone https://github.com/ZrrSkywalker/MonoDETR.git
    cd MonoDETR

    conda create -n monodetr python=3.8
    conda activate monodetr
    ```

2. Install pytorch and torchvision matching your CUDA version:
    ```bash
    conda install pytorch torchvision cudatoolkit
    # We adopt torch 1.9.0+cu111
    ```

3. Install requirements and compile the deformable attention:
    ```
    pip install -r requirements.txt

4. To compile the deformable attention, you need to install the `torch` and `torchvision` first. Then run the following commands:
    ``` 
    cd lib/models/monodetr/ops/
    bash make.sh

    cd ../../../..
    ```

5. Make dictionary for saving training losses:
    ```
    mkdir logs
    ```

6. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```
    │MonoDETR/
    ├──...
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    │   ├──training/
    │   ├──testing/
    ├──...
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monodetr.yaml`.

## Get Started

### Train
You can modify the settings of models and training in `configs/monodetr.yaml` and indicate the GPU in `train.sh`:

    bash train.sh configs/monodetr.yaml > logs/monodetr.log

### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monodetr.yaml`:

    bash test.sh configs/monodetr.yaml

To enable the profiler (deepspeed) set `profile: True` in the tester section of the config file.


## Acknowlegment
This repo build on top of the original [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR)
