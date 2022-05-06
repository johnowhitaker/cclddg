# CLOOB Conditioned Latent Denoising Diffusion GAN
> My code and utilities for training CCLDDGs.


This is very much a work in progress. Stay tuned for better info soon :)

## Install

At the moment I'd suggest cloning this and adding it to your path.

## What is all this

The main thing this code does is define a UNet architecture and an accompanying Discriminator architecture that can take in an image (or a latent representation of one) along with conditioning information (what timestep we're looking at, a CLOOB embedding of an image or caption) and a latent variable `z` used to turn the unet into a more GAN-like multimodal generator thingee. 

Coming soon, demos of this as
- A standard diffusion model
- A standard latent diffusion model
- A standard Defusion Denoising GAN
- A latent Defusion Denoising GAN
- CLOOB-Conditioned Latent Defusion Denoising GAN
- Training a text-to-image model with no text

```python
# for now here's a sum
3+5
```




    8



# Running the training script

This script is written to run OUTSIDE this directory (aka NOT in cclddg). It also assumes the locations of various dependancies and model files. To set it up, in a notebook run:

```python
# !git clone https://github.com/johnowhitaker/cclddg                               &>> install.log
# !git clone https://github.com/CompVis/latent-diffusion                           &>> install.log
# !git clone https://github.com/CompVis/taming-transformers                        &>> install.log
# !pip install -e ./taming-transformers                                            &>> install.log
# !git clone --recursive https://github.com/crowsonkb/cloob-training               &>> install.log
# !git clone https://github.com/openai/CLIP/                                       &>> install.log
# !pip install CLIP/.                                                              &>> install.log
# !pip install --upgrade webdataset ipywidgets                                     &>> install.log
# !pip install datasets omegaconf einops wandb pytorch_lightning                   &>> install.log
# !wget https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                     &>> install.log
# !unzip -q kl-f8.zip  
```

Then if you wish to use W&B for logging, run `wandb login` in a terminal.

Then copy the script from the cclddg folder downloaded as part of the command above to your local dir:

`cp cclddg/train_cclddg.py train.py`

And run your training like so:

`python train.py --z_dim 16 --n_channels_unet 64 --batch_size 64 --n_batches 20 --lr_gen 0.0001 --lr_disc 0.0001 --log_images_every 50 --save_models_every 500 --n_steps 8 --dataset celebA --wandb_project cclddg_faces`
