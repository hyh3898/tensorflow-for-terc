# Multi-label-Inception-net
Modified `retrain.py` script to allow multi-label image classification using pretrained [Inception net](https://github.com/tensorflow/models/tree/master/inception).

The `label_image.py` has also been slightly modified to write out the resulting class percentages into `results.txt`. 

Replace new softmax activation function with sigmoid,
Check [here](https://medium.com/@bartyrad/multi-label-image-classification-with-inception-net-cbb2ee538e30) for detailed explanation of all the changes and reasons



# Overview

This repo contains code for the "TensorFlow for poets 2" series of codelabs.

There are multiple versions of this codelab depending on which version 
of the tensorflow libraries you plan on using:

* For [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) the new, ground up rewrite targeted at mobile devices
  use [this version of the codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite) 
* For the more mature [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro) use 
  [this version of the codealab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).


This repo contains simplified and trimmed down version of tensorflow's example image classification apps.

* The TensorFlow Lite version, in `android/tflite`, comes from [tensorflow/contrib/lite/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite).
* The Tensorflow Mobile version, in `android/tfmobile`, comes from [tensorflow/examples/android/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).

The `scripts` directory contains helpers for the codelab. Some of these come from the main TensorFlow repository, and are included here so you can use them without also downloading the main TensorFlow repo (they are not part of the TensorFlow `pip` installation).



# Processing Data

* [Data for training and validation](https://drive.google.com/open?id=1Iv2JrOl-8XocuUb0TUdN0V90T46qa5Mk)

# Preprocessing Data
### Dataset
* Provided by Terc
* Scraped from Windows on Earth website Briefly describe how you scraped, e.g. package you used
* Link to [downloadable data](https://drive.google.com/open?id=1Iv2JrOl-8XocuUb0TUdN0V90T46qa5Mk)

# Usage
### Retrain the Model
```shell
$ ARCHITECTURE='inception_v3'
```
Retrain:
```shell
$ ARCHITECTURE='inception_v3'
$ python -m scripts.retrain \
	--image_dir=tf_files/woe/woe_photos \
	--output_graph=tf_files/woe/retrained_graph.pb \
	--output_labels=tf_files/woe/retrained_labels.txt \
	--summaries_dir=tf_files/woe/training_summaries/"${ARCHITECTURE}" \
	--how_many_training_steps=500 \
	--learning_rate=0.01 \
	--testing_percentage=10 \
	--validation_percentage=10 \
	--model_dir=tf_files/models/ \
	--bottleneck_dir=tf_files/woe/woe_bottlenecks \
	--architecture="${ARCHITECTURE}" \
	--test_batch_size=-1 \
	--validation_batch_size=-1
```

Visualize with TensorBoard:
```shell
$ tensorboard --logdir tf_files/training_summaries &
```

If you want to kill all existing TensorBoard instances:
```shell
pkill -f "tensorboard"
```

### Predict the new Images with retrained model
```shell
python scripts/label_image.py \
  --graph=tf_files/woe/woe_retrained_graph.pb \
  --labels=tf_files/woe/woe_retrained_labels.txt \
  --image=tf_files/woe/woe_photos/class/photo_name.jpg

```
