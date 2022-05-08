# CLOOB Conditioned Latent Denoising Diffusion GAN
> My code and utilities for training CCLDDGs.


This is very much a work in progress. Stay tuned for better info soon :)

In the meantime, the core ideas:

- Diffusion models gradually noise an image and learn the reverse process. Great but lots of steps means slow inference
- DDGs tweak the training to include a discriminator, and promise good results in very few steps at inference
- Latent diffusion means doing it in the latent space of some autoencoder, which is more efficient.
- We can condition these models with various extra info. I chose to use CLOOB embeddings as conditioning info. That way we can get these from images or text or both during training, and at inference feed in either text or images and use them as conditioning for the generation process. In theory this gives a nice way to do text-to-image!

I tried training for a few hours on CC12M and while the results aren't photorealistic you can definitely see *something* of a prompt in the outputs - 'blue ocean waves' is mostly blue, for eg. Results from a longer trainging run soon.

## Install

I might turn this into a package later, for now your best bet is to check out the colab(s) below or follow the instructions in 'Running the training script'.

## What is all this

The main thing this code does is define a UNet architecture and an accompanying Discriminator architecture that can take in an image (or a latent representation of one) along with conditioning information (what timestep we're looking at, a CLOOB embedding of an image or caption) and a latent variable `z` used to turn the unet into a more GAN-like multimodal generator thingee. 

Demos
- A standard diffusion model TODO
- A standard latent diffusion model TODO
- A standard Defusion Denoising GAN TODO
- CLOOB-Conditioned Latent Defusion Denoising GAN: https://colab.research.google.com/drive/1T5LommNOw4cVr8bX6AO5QXJ7D1LyXz2m?usp=sharing (faces)

W&B runs TODO

# Running the training script

The train script is written to run OUTSIDE this directory (aka NOT in cclddg). It also assumes the locations of various dependancies and model files. To set it up, in a notebook run:

```python
# !git clone https://github.com/johnowhitaker/cclddg                               &>> install.log
# !git clone https://github.com/CompVis/latent-diffusion                           &>> install.log
# !git clone https://github.com/CompVis/taming-transformers                        &>> install.log
# !pip install -e ./taming-transformers                                            &>> install.log
# !git clone --recursive https://github.com/crowsonkb/cloob-training               &>> install.log
# !git clone https://github.com/openai/CLIP/                                       &>> install.log
# !pip install CLIP/.                                                              &>> install.log
# !pip install --upgrade webdataset ipywidgets lpips                               &>> install.log
# !pip install datasets omegaconf einops wandb pytorch_lightning                   &>> install.log
# !wget https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                     &>> install.log
# !unzip -q kl-f8.zip 
```

Then if you wish to use W&B for logging, run `wandb login` in a terminal.

Then copy the script from the cclddg folder downloaded as part of the command above to your local dir:

`cp cclddg/train_cclddg.py train.py`

And run your training like so:

`python train.py --z_dim 16 --n_channels_unet 64 --batch_size 64 --n_batches 200 --lr_gen 0.0001 --lr_disc 0.0001 --log_images_every 50 --save_models_every 500 --n_steps 8 --dataset celebA --wandb_project cclddg_faces`

# More info

Let's break this down into bits

## Diffusion models

Link DDPM paper

Image

Demo using this code

## Denoising Diffusion GANs

Link paper

Diagram

Explanation

How this differs from paper (read paper)

Demo using this code

## Latent Diffusion

Explain AE

Latent DDG demo

## CLOOB Conditioning

Explain

Demo

Examples of text-to-image capacity trained on just images
