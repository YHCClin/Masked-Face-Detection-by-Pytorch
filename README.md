# Masked Face Detection



## Some Prepare

&emsp;Please make sure the following packages are included in your environment.

* python 3.7.4
* pytorch 1.4.0
* tqdm 4.36.1
* tensorboardX 2.1

* Pillow 7.0.0
* opencv 3.4.2



## How To setup and run the project

### Train

Execute the following commands in the project root directory

```
python train_model.py --epoch <the epoch number you want to train>
```

You can also execute the following command to view the help

```
python train_model.py -h
```



## Test

We provide three test modes, including getting images from the camera, getting images from files, and simply running from the test set to get the accuracy.

#### video

```
python test_model.py --mode video
```

#### Image File

```
python test_model.py --mode image --filepath <your_filepath>
```

#### Image DataSet

```
python t4st_model.py --mode imageset
```



