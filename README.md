# pix2pix : The shoes implementation

The original algorithm this project dependents on is [Image-to-Image Translation with Conditional Adversarial Nets](https://arxiv.org/abs/1611.07004) which is usually called pix2pix. This image translation paper applys CGAN(conditional 
genenrative adversarial network) with the semantic labels figs as the input while the realistic photos regarded as the 
object figs.

* [The author of pix2pix ](https://github.com/phillipi/pix2pix) provide the torch implementation of CGAN with U-skip.
* Another engineer give the [tensorflow implementiation](https://github.com/affinelayer/pix2pix-tensorflow) of this project and write a [website](https://affinelayer.com/pix2pix/) describing how pix2pix
works.

This project draw a simple CGAN without U-skip and paches-GAN, and gets well results due to the power of CGAN. I fix attention on 
the shoes dataset including 4 main categories: boots, sandals, shoes, slippers. The original dataset called ut-zap50k can be found [here](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/). The egdes of these shoes is drawn by sobel filters which can be 
used through scipy.


## Prerequirites

- python3.6 (python2.7 also OK with small changes)
- tensorflow 1.4+ (This project is compatibal with tensorflow1.8)
- numpy
- PIL
- imageio
- (optional) scipy
- (optional) matplotlib


## Usage

This project provide a small dataset containing 1000 images of shoes and its edge labels, for quick implementation. You can apply your own dataset if you like, but pay attention on the model size. The recomended size of figs my architecture welcomes is 128*128, the details you can imitate the toy dataset I provide.

First, unzip the train.zip which is under the data dir. Make sure the data dir path is right for the model, the part processing dataset is in the utils.py, because the unzip way in Windows and Ubuntu is different. This is a small problem, for I don't apply flags function.

Second, the training part is in the main.py, including the epochs. Not many parameters you need to decide by yourself, because this project is a quick implementation. Just run main.py and you can get the results under the sample dir.

Third, no third, that's all.


## Extended Part

Inception score now available under the tool dir, you can also draw the gif of your results like the gif shown in this readme.me. The simplest way is to copy the gif.py and place it in the same path of main.py, and just run it in spyder or cmd or whatever you run your python file.

## Good lucks 

Wish you enjoy your study with deep learning and GAN.


## Email
You can contact me by emailing : leigaoyi@126.com 

