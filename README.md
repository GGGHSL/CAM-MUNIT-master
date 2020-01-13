# CAM-MUNIT
#### An unsupervised selfie-to-anime translation project: using GAN with attention.

> **Abstruct**   
_On the cross-domain image translation task, GANs usually consist of three parts: the encoder(s) and the decoder(s), 
together as a generator, to map between the latent space and the data space, and a discriminator to put an adversarial 
constraint on the latent and/or the data space._
>
>_Previously, there are typically two kinds of GAN structures for this problem (selfie-to-anime face translation problem):_ 
> * _Include two encoders (one for content code and the other for style code) and one decoder (take MUNIT [Huang, 2018] 
and WarpGAN [Shi, 2019] for example)_
> * _Includes only one encoder (take CartoonGAN [Chen, 2018] for instance)_
>
>_On this task, MUNIT produced more diverse colors and patterns, possibly because it requires the decoder to produce all 
the new pixels from the highly compressed feature encoding, which may lose the details of the original image at the same 
time and the content of the generated image may not match the original very well. CartoonGAN, on the other hand, uses 
the pixels directly from the input image, so the content can match better. Hewer, this method may make the generated 
image too realistic for the original details, and, for example, it may make the pixels in the forehead area the same 
color as those in the hair areas._
>
>_In this work, the encoders, decoders and discriminators are the same structure as MUNIT, but to solve the problems 
mentioned above, an extra attention module is added to the content code and the input of the discriminator. 
In addition, an extra perceptual loss, calculated by a pretrained VGG19 model, is added to keep the content of the 
generated image unchanged. As for the training strategy, I first pretrained the generator using the content loss only 
as the initialization phase, considering that random initialization is likely to produce less dispersed samples which 
may lead to mode collapse._

#### Examples
>Every two rows is a group: the first line is the source, the second line is the generated animation image.

![examples](https://github.com/GGGHSL/CAM-MUNIT-master/blob/master/examples/test_examples.jpg?raw=true)

#### Requirements
>* python == 3.7
>* pytorch == 1.1.0

#### Usage
```
├── datasets
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

#### Train
```bash
$ python train_multigpus.py --gpus_ids=YOUR_GPU_IDS --output_path=YOUR_SAVE_PATH --phase=main-train --resume=False
```

#### Test
```bash
$ python test.py --load_path=YOUR_MODEL_PATH --iteration=IF_NEED_TO_SPECIFY
```
