# DeepSim

This is a tensorflow implementation of the paper [Generating Images with Perceptual Similarity Metrics based on Deep Networks](https://arxiv.org/abs/1602.02644) by Alexey Dosovitskiy, Thomas Brox.

This repo is based on CharlesShang's work [TFFRCNN](https://github.com/CharlesShang/TFFRCNN). I really appreciate their great work.

I mainly use the data load module of their work in `./deepSimGAN/util.py`. You can remove all codes outside the `./deepSimGAN` directory if you rewrite the DataFetcher class in the `./deepSimGAN/util.py` script.

### requirement

- python 2.7
- tensorflow >= 1.1.0
- python-opencv >= 3.2.0
- numpy >= 1.11.3
- tqdm

## Training

To train your own deepsim model, you need to:
1. Prepare dataset and pretrained-model for encoder training.
2. Train your encoder and save the fine-tuned checkpoint.
3. Prepare dataset for generator and discriminator training.
4. Load fine-tuned encoder and train the generator and discriminator.

### Prepare dataset and pretrained model for encoder

1. Download the training, validation, data and VOCdevkit to the target directory named `VOCdevkit`, such as `/data/VOCdevkit`. We use `$VOCdevkit` to refer to it and use `$DeepSim` to refer to this repo's root directory.

    ```Shell
    cd $VOCdevkit
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
    ```

2. Extract all of these tars

    ```Shell
    tar xvf VOCtrainval_11-May-2012.tar
    tar xvf VOCdevkit_18-May-2011.tar
    ```

3. It should have this basic structure

    ```Shell
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2012                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```
4. Create symlinks for the PASCAL VOC dataset

    ```Shell
    cd $DeepSim/data
    ln -s $VOCdevkit VOCdevkit2012
    ```

5. Download pre-trained model [VGG16](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path `$DeepSim/data/pretrain_model/VGG_imagenet.npy`


### Train the encoder

The codes of encoder net deninition and training are in `deepSimGAN/EncoderNet.py`. You can use following command to start your training:

```Shell
cd $DeepSim
python deepSimGAN/EncoderNet.py --weight_path data/pretrain_model/VGG_imagenet.npy --logdir output/encoder
```

All of checkpoints and summaries are stored in the given logdir. You can use tensorboard to monitor the training process

```Shell
tensorboard --logdir output/encoder --host 0.0.0.0 --port 6006
```

### Prepare dataset for generator and discriminator training

You can just reuse the Pascal VOC 2012.

If you want to use other datasets, remember to **rewrite the class DataFetcher in `./deepSimGAN/util.py`**


### Train your deepSimNet

The code of deepSimNet definition is in `deepSimGAN/deepSimNet.py` and the code of training is in `deepSimGAN/main.py`. You can use following command to start your training:

```Shell
python deepSimGAN/main.py --encoder output/encoder --logdir output/deepsim
```

There are many other arguments which can be specified to influence your training. Please refer to the argument parser in `deepSimGAN/main.py`.


