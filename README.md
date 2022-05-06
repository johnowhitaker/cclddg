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

```
# for now here's a sum
3+5
```




    8


