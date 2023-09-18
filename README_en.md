# A demonstration of automotive interior inspection based on yolov5



## Technical Program

1. YOLOv5 is utilized to train the target detection model (steering wheel, passenger front flap) for target detection and box crop for image or video frame extraction.

2. The box crop is converted to RGB format and then converted to an array, and then, K-Means clustering is performed to determine the predominant color within the region.

3. Compare the color with the correct color and output the result.




## Project deployment

### Environment Configuration

Creating the virtual environment

```
conda create -n yolo python==3.8.5
conda activate yolo
```



Install `PyTorch`

```
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # For 30 series or higher gpu
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # For 10/20/mx series
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # For cpu
```



Install requirements

```
pip install -r requirements.txt
```



### Model Training

Install `labelimg`

```
pip install labelimg
```



Run `labelimg` to label the image

```
labelimg
```

Note that the label save path is modified, and also that the **output format should be changed to YOLO format**.



Modify the configuration file `my_data.yaml`

```
train: ./datasets/images/train/  # Path of the training set
val: ./datasets/images/test/  # Path of the test set

nc: 2  # Number of classes

names: ['wheel', 'panel']  # Names of classes
```



Run command

```
python train.py --data my_data.yaml --cfg yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 50 --batch-size 4
```

The results will be saved in `./runs/train`



### Prediction using the model

Modify the `line 43 `in `window_main.py`

```
self.model = self.model_load(weights="runs/train/exp9/weights/best.pt",
```

Change it to the path of the model you want to use, the `best.pt` and `last.pt` in the weights folder are the best model in training and the last epoch model respectively



Run `window_main.py`



### Precaution

If you have problems reading the pictures, please downgrade the pillow version to 8.4.0.

```
pip install pillow==8.4.0
```



If `numpy` reports an error `module 'numpy' has no attribute 'int'`, please downgrade the version of numpy.



## Problems and directions for improvement

- Because the dataset is too small and because the photos in the dataset have different models and vary in light and color temperature, it may results in low detection accuracy.

- Practical use with fixed light and color temperatures, fixed models, and a larger amount of data will run a more accurate model.
- Error correction can be added so that if an error is made during early use, it can be corrected manually, which can further train the model and make it more accurate in use.



## NOTE

- During the actual model training process, attention should be paid to taking photos from multiple angles.

- During the actual shooting process, attention should be paid to the issue of blurring.




## Reference

1. yolov5, https://github.com/ultralytics/yolov5
2. YOLOV5-sfid, https://gitee.com/song-laogou/yolov5-sfid
3. Dataset: Persian Car Interior Design, https://www.kaggle.com/datasets/behnamhasanbeygi/persian-car-interior-design



![](https://github.com/BromineBr/yolov5-automotive-interior-inspection/blob/main/images/UI/waifu.png)

Produced by Bromine_Br & Hina Kagiyama
