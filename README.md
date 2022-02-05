# Machine Learning Model Applied on Chest X-Ray Images 
## Project purpose
It is an automatic classification of COVID-19 images pneumonia from other pneumonia and typical x-ray images. Project made for "Jetson AI Specialist" certification project. Build and submit as an open-source project that uses NVIDIA Jetson and elements of AI and ML with GPU acceleration. Project Type: Image Classification.

## Equipment & software used
NVIDIA Jetson Nano 2GB Developer Kit<br>
https://www.nvidia.com/pl-pl/autonomous-machines/embedded-systems/jetson-nano/

JetPack 4.6<br>
https://developer.nvidia.com/embedded/jetpack

## Dataset Overview
Two datasets were used in this project; both were merged in one.

#### COVID CXR Image Dataset (Research)
source: https://www.kaggle.com/sid321axn/covid-cxr-image-dataset-research <br>
This dataset consists of 1823 images of view of Chest X-ray images. Labelled Optical Coherence Tomography(OCT) and CXR Images used for viral pneumonia and non-pneumonia or typical cases.

#### Covid-19 Image Dataset
source: https://www.kaggle.com/sid321axn/covid-cxr-image-dataset-research <br>
The dataset consists of 137 images of COVID-19 and 317 in total containing Viral Pneumonia and Normal Chest X-Rays.

# Project steps

#### 1. Preparing enviroment<br>
Jetson Nano environment prepared according to "Getting Started with Jetson Nano 2GB Developer Kit" tutorial.
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit

On Jetson Nano 2GB, it is necessary to disable ZRAM and create a swap file (4GB) as training uses up a lot of extra memory.
https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md#mounting-swap

#### 2. Data Preparation<br>
I merged both datasets into one. I divided files from both datasets into the structure presented below:<br>
- test ~80%, 
- train ~10%, 
- val ~10%  of total files.
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

#### 3. Model training<br>

Launching the Container<br>
```ruby
docker/run.sh
```
<br>
Model training and exporting to ONNX (40 epochs)
<br>

```ruby
jetson-inference/python/training/classification# python3 train.py --model-dir=models/Covid19 --batch-size=4 --workers=1 --epochs=40 data/Covid19
jetson-inference/python/training/classification# python3 onnx_export.py --model-dir=models/Covid19
```

![image](https://user-images.githubusercontent.com/67101428/151599709-b203f805-3c1c-4e49-941c-8c65a389b13a.png)

Accuracy achieved: 83.3% which is a promising performance but yet to be further improved.
<br>

#### 4. Checking model<br>
Let's check the model's effectiveness using independent images from three categories (covid, typical, virpneumonia).
```ruby
imagenet --model=models/Covid19/resnet18.onnx --labels=data/Covid19/labels.txt --input_blob=input_0 --output_blob=output_0 data/Covid19/test_input_covid/ data/Covid19/test_output/
imagenet --model=models/Covid19/resnet18.onnx --labels=data/Covid19/labels.txt --input_blob=input_0 --output_blob=output_0 data/Covid19/test_input_normal/ data/Covid19/test_output/
imagenet --model=models/Covid19/resnet18.onnx --labels=data/Covid19/labels.txt --input_blob=input_0 --output_blob=output_0 data/Covid19/test_input_virpneumonia/ data/Covid19/test_output/
```

![c3](https://user-images.githubusercontent.com/67101428/151595988-d4e4fbfc-718c-4e25-9dea-fc359947739c.jpg)
![n3](https://user-images.githubusercontent.com/67101428/151596037-5c3e19e3-6f49-4f94-8cc7-fbf8b050844e.jpg)
![v1](https://user-images.githubusercontent.com/67101428/151596092-8e2ca207-cc06-4fcf-be31-498ead116268.jpg)

## Video
https://youtu.be/BONkoj0dQfI

## Reference
https://github.com/dusty-nv/jetson-inference/
