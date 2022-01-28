# Classification of Chest X-ray images

Made for Jetson AI Specialist certification project. Build and submited an open-source project that uses NVIDIA Jetson and elements of AI 
(machine learning) with GPU acceleration. Project Type: Image Classification.

## Equipment & software used
NVIDIA Jetson Nano 2GB Developer Kit
https://www.nvidia.com/pl-pl/autonomous-machines/embedded-systems/jetson-nano/

JetPack 4.6
https://developer.nvidia.com/embedded/jetpack

## Dataset Overview
For the project two datasets were used, both dataset was merged in one.

#### COVID CXR Image Dataset (Research)
source: https://www.kaggle.com/sid321axn/covid-cxr-image-dataset-research
This dataset consists of 1823 images of an annotated posteroanterior (PA) view of Chest X-ray images. Labeled Optical Coherence Tomography(OCT) and CXR Images used for viral pneumonia and non-pneumonia or normal cases.

#### Covid-19 Image Dataset
source: https://www.kaggle.com/sid321axn/covid-cxr-image-dataset-research
It contains around 137 cleaned images of COVID-19 and 317 in total containing Viral Pneumonia and Normal Chest X-Rays structured into the test and train directories.

# Project steps

#### 1. Preparing enviroment<br>
According to "Getting Started with Jetson Nano 2GB Developer Kit" tutorial
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit

Mounting Swap
https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md#mounting-swap

#### 2. Data Preparation<br>
Merging data in one dataset. I divided files from both datasets into structure presented below.
test ~80% of files, train ~10% of files, val ~10% of files.
<br><br>
<b>Covid19</b><br>
├ test<br>
│  ├ Covid<br>
│  ├ Normal<br>
│  └ Viral Pneumonia<br>
│<br>
├ train<br>
│  ├ Covid<br>
│  ├ Normal<br>
│  └ Viral Pneumonia<br>
│<br>
├ val<br>
│  ├ Covid<br>
│  ├ Normal<br>
│  └ Viral Pneumonia<br>
│<br>
└ label.txt<br>
<br>

#### 3. Re-training ResNet-18 Model<br>

Starting the container
```ruby
docker/run.sh
```
<br>
Model training and exporting to onnx (40 epochs)
<br>

```ruby
jetson-inference/python/training/classification# python3 train.py --model-dir=models/Covid19 --batch-size=4 --workers=1 --epochs=40 data/Covid19
jetson-inference/python/training/classification# python3 onnx_export.py --model-dir=models/Covid19
```

![image](https://user-images.githubusercontent.com/67101428/151508491-9f035372-787a-4fa8-b3ae-c13786044b4d.png)

Accuracy achieved: 83.3% which is a promising performance but yet to be further improved.
<br><br>


#### 4. Running model<br>
root@jetson:/jetson-inference/python/training/classification# imagenet --model=models/Covid19/resnet18.onnx --labels=data/Covid19/labels.txt --input_blob=input_0 --output_blob=output_0 data/Covid19/test/ data/Covid19/test_output/

![image](https://user-images.githubusercontent.com/67101428/151514710-781c9c44-d4a7-4f24-8730-02bc59cd6eca.png)


## Video
https://youtu.be/yprSOHy7r74


## Reference
Inspired by https://github.com/dusty-nv/jetson-inference/
