# auto-medical-detection
An implementation of a framework for automatic adaption medical image detection which based on segmentation data.

## Overview
Major Features:
- Automated analysis of the dataset, automatic adaptation to new medical segmentation datasets without user intervention.
- Automatically designs and executes a network training pipeline. 
- Implementations of prevalent object detectors: e.g. 2D and 3D U-Net, Mask R-CNN, Retina Net, Retina U-Net. 
- Modular and light-weight structure for backbone architecture: e.g. resnet, densenet, inceptionResNetV2.
- Dynamic patching and tiling (for training and inference) or full-size images.
- Weighted consolidation of box predictions across patch-overlaps, ensembles.
- Pipeline on data analysis, preprocessing, training, postprocessingg, evaluation, inference and visualization.
- Automatically chooses the best single model or ensemble to be used for test set prediction.


## Code Structure
#### Common structure
  - [Data analysis]: DatasetAnalyzer, Planner, Planner2D, 
  - [Datasets]: DataLoaderBase, AbstractAugmentation, BatchGenerator2D, BatchGenerator3D
  - [Preprocessing]: GenericPreprocessor, PreprocessorFor2D
  - [Training]: GenericTrainer, Trainer, CascadeTrainer
  - [Inference]:Predictor
  - [Evaluation]: Evaluator
  - [Models]: models/*
  - [Utilities]: utils/*
  - [Bin]: bins/*


## Installation
1. Clone this repository
2. Setup package in virtual environment
    ```
    cd nodule_detector
    virtualenv -p python3 venv
    source venv/bin/activate
    pip3 install -e .
    ```
3. Install dependencies
   ```bash
   pip3 install -r requirements.txt
