# Prepare:
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

# Assumes you're not in the cclddg directory so cp this to parent dir where the model above is 
# downloaded (along with the other github repos)

import sys
sys.path.append('cclddg')
sys.path.append('./latent-diffusion')
sys.path.append('./cloob-training')
from cclddg.core import UNet, Discriminator
from cclddg.ddg_context import DDG_Context
from cclddg.data import get_paired_vqgan, tensor_to_image
from cloob_training import model_pt, pretrained
import ldm.models.autoencoder
from omegaconf import OmegaConf
from PIL import Image
import torch
import numpy as np
import webdataset as wds
import torch.nn as nn
import torchvision.transforms as T
from torch.nn import functional as F
from torchvision.utils import make_grid
import numpy as np
import itertools
from tqdm import tqdm
import pprint


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# CLOOB setup
print('Setting up CLOOB')
scale_224 = T.Resize(224)
config = pretrained.get_config('cloob_laion_400m_vit_b_16_16_epochs')
cloob = model_pt.get_pt_model(config)
checkpoint = pretrained.download_checkpoint(config)
cloob.load_state_dict(model_pt.get_pt_params(config, checkpoint))
cloob.eval().requires_grad_(False).to(device)
print('Done')

# Load the autoencoder
print('Loading Autoencoder')
ae_model_path = 'model.ckpt'
ae_config_path = 'latent-diffusion/models/first_stage_models/kl-f8/config.yaml'
ae_config = OmegaConf.load(ae_config_path)
ae_model = ldm.models.autoencoder.AutoencoderKL(**ae_config.model.params)
ae_model.eval().requires_grad_(False).to(device)
ae_model.load_state_dict(torch.load(ae_model_path)['state_dict'])
print('Done')

def train(args):
    
    # Init logging
    if args.wandb_project != 'None':
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Set up the data
    if args.dataset == 'vqgan_pairs':
        data = get_paired_vqgan(batch_size=args.batch_size)
    else:
        print('Dataset not recognized')
        return 0
    data_iter = iter(data)

    # Create the model
    unet = UNet(image_channels=4*2, n_channels=args.n_channels_unet,
               z_dim=args.z_dim, n_z_channels=args.n_z_channels,
               ).to(device) # 4 or 8 or whatever based on ae
    disc = Discriminator(image_channels=4*3, n_channels=args.n_channels_disc).to(device) # image_channels=4 but expects 8 since we condition on xt

    # Set up the DDG context
    ddg_context = DDG_Context(n_steps=args.n_steps, beta_min=args.beta_min, 
                              beta_max=args.beta_max, device=device)



    criterion = nn.BCELoss().to(device)

    losses = [] # Store losses for later plotting

    optim_gen = torch.optim.AdamW(unet.parameters(), lr=args.lr_gen, weight_decay=args.weight_decay) # Optimizer
    optim_dis = torch.optim.AdamW(disc.parameters(), lr=args.lr_disc, weight_decay=args.weight_decay) # Optimizer

    # Code for storing examples
    def examples(n_examples = 5, z_dim=8, img_size=256):
        im_out = Image.new('RGB', (img_size*n_examples, img_size*3))
        lq, hq = next(data_iter)
        lq = lq_tfm(lq[:n_examples]).to(device)*2-1
        hq = hq_tfm(hq[:n_examples]).to(device)*2-1
        for i in range(n_examples):
            im_out.paste(tensor_to_image(lq[i]), (i*img_size, 0))
            im_out.paste(tensor_to_image(hq[i]), (i*img_size, img_size))
        # return im
        batch_size=lq.shape[0]
        cond_0 = ae_model.encode(lq).mode()
        z = torch.randn((n_examples,z_dim), device=device)
        # c = cloob.text_encoder(cloob.tokenize([p]*n_examples).to(device)).float()
        batch = cloob.normalize(scale_224(lq.cuda()))
        c = cloob.image_encoder(batch).float()

        # Starting from random x
        x = torch.randn(n_examples, 4, img_size//8, img_size//8).to(device) # TODO from img_size
        t = torch.ones((n_examples,), dtype=torch.long).to(device)*ddg_context.n_steps
        while t[0] > 0:
            gen_input = torch.cat((x, cond_0), dim=1)
            pred_im = unet(gen_input, t, c, z)[:,:4,:,:]
            x, n = ddg_context.q_xt_x0(pred_im, t-1)
            t -= 1
            if t[0]==0:
                for s in range(n_examples):
                    im_out.paste(ddg_context.tensor_to_image(ae_model.decode(pred_im[s].unsqueeze(0))), (img_size*s, img_size))
        return im_out

    def log_examples():
        im = examples(n_examples=5, z_dim=args.z_dim, img_size=args.img_size)
        if args.wandb_project != 'None':
            wandb.log({'Reconstruction':wandb.Image(im)})
    
    # Transform lq and hq TODO use img_size from args and add other args
    # Goal is 4x SR. If image size is 256 (hq) we take 128px from lq (which is already 1/2 res) and scale to 64px then back up to 256
    lq_tfm = T.Compose([T.CenterCrop(args.img_size//2), T.Resize(args.img_size//4), T.Resize(args.img_size)])
    hq_tfm = T.CenterCrop(args.img_size)

    for i in tqdm(range(0, args.n_batches)): # Run through the dataset

        # Get a batch
        lq, hq = next(data_iter)
        lq = lq_tfm(lq).to(device)*2-1
        hq = hq_tfm(hq).to(device)*2-1
        batch_size=lq.shape[0]

        log = {} # Prepare logging

        unet.zero_grad() # Zero the gradients
        disc.zero_grad() # Zero the gradients

        # Prepare data
        x0 = ae_model.encode(hq).mode() # To latents
        cond_0 = ae_model.encode(lq).mode()
        t = torch.randint(1, ddg_context.n_steps, (batch_size,), dtype=torch.long).to(device) # Random 't's 
        z = torch.randn((batch_size,args.z_dim), device=device) # It's a gan now yay
        
        # Get cloob embeddings TODO check if pct_text is 0 or 1 to save time
        with torch.no_grad():
            batch = cloob.normalize(scale_224(lq.cuda()))
            c = cloob.image_encoder(batch).float()
        
        # zero out 10%:
        zero_mask = (torch.cuda.FloatTensor(batch_size, 1).uniform_() < 0.9).float().expand(-1, 512)
        c = c*zero_mask

        # Get the noised images (xt) and the noise (our target) plus the x(t-1)
        xtm1, eps_t = ddg_context.q_xt_x0(x0, t-1) # Most of the noise
        xt, eps_added = ddg_context.q_xt_xtminus1(xtm1, t) # One extra step

        # Disc loss on the 'real' samples
        disc_input = torch.cat((xt, xtm1, cond_0), dim=1)
        disc_pred_real = disc(disc_input, t, c) # Predict for xtm1 conditioned on xt
        label = torch.ones_like(disc_pred_real) 
        disc_loss_real = criterion(disc_pred_real, label)
        disc_loss_real.backward()
        log['disc_loss_real'] = disc_loss_real.item()
        log['D(real).mean()'] = disc_pred_real.mean().item()

        # Disc on a fake batch
        gen_input = torch.cat((xt, cond_0), dim=1) # We use the lq image as conditioning
        gen_pred_x0 = unet(gen_input, t, c, z)[:,:4,:,:] # Run xt through the network to get its predictions
        gen_pred_noised_to_xtm1, noise_added_to_pred = ddg_context.q_xt_x0(gen_pred_x0, t-1)
        label.fill_(0) # Change labels to 0
        disc_pred_fake = disc(torch.cat((xt, gen_pred_noised_to_xtm1.detach(), cond_0), dim=1), t, c) # detach to avoid going through graph twice
        disc_loss_fake= criterion(disc_pred_fake, label)
        disc_loss_fake.backward()
        log['disc_loss_fake'] = disc_loss_fake.item()
        log['D(fake).mean()'] = disc_pred_fake.mean().item()
        optim_dis.step() # Update the discriminator 
        disc.zero_grad() # Zero out grads again (can also put in eval mode)

        ## TRAIN THE GENERATOR PART
        # Disc loss:
        unet.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        disc_pred_fake_G = disc(torch.cat((xt, gen_pred_noised_to_xtm1, cond_0), dim=1), t, c)
        disc_loss_fake_G = criterion(disc_pred_fake_G, label)
        disc_loss_fake_G.backward(retain_graph=True) # We also want to update based on recon loss
        log['disc_loss_fake_G'] = disc_loss_fake_G.item()
        log['D(fake).mean() G'] = disc_pred_fake_G.mean().item()

        # Reconstruction loss?
        # TODO add in or leave out?
        l = F.mse_loss(x0.float(), gen_pred_x0) # Compare the predictions with the targets
        log['recon_loss'] = l.item()
        recon_loss = args.recon_loss_scale*l
        recon_loss.backward()

        optim_gen.step() # Update the discriminator 

        losses.append(log)
        if args.wandb_project != 'None':
            wandb.log(log)

        # Occasionally log demo images
        if (len(losses))%args.log_images_every==0:
            log_examples()
            
        # Occasionally save models
        if (len(losses))%args.save_models_every==0:
            torch.save(unet.state_dict(), f'unet_ar_{len(losses):06}.ckpt')
            torch.save(disc.state_dict(), f'disc_ar_{len(losses):06}.ckpt')
            
        # TODO add a CLOOB metric for text-to-image runs

    if args.wandb_project != 'None':
        wandb.finish()
    
import argparse
parser = argparse.ArgumentParser(description='Train CCLDDG (SR/AR)')
parser.add_argument('--n_batches',type=int, default=10, help='How many batches should we train on')
parser.add_argument('--dataset',type=str, default='vqgan_pairs', help='What dataset? only vqgan_pairs supported')
parser.add_argument('--lr_gen',type=float, default=1e-4, help='LR for unet')
parser.add_argument('--lr_disc',type=float, default=4e-5, help='LR for discriminator')
parser.add_argument('--batch_size',type=int, default=32, help='batch size')
parser.add_argument('--img_size',type=int, default=128, help='image resolution')
parser.add_argument('--wandb_project',type=str, default='None', help="Leave as 'None' if you don't want to log to W&B")
parser.add_argument('--log_images_every',type=int, default=50, help='How frequently log ims')
parser.add_argument('--save_models_every',type=int, default=500, help='Save models')
parser.add_argument('--z_dim',type=int, default=8, help='z dim for unet')
parser.add_argument('--n_z_channels',type=int, default=16, help='n z channels for unet')
parser.add_argument('--n_channels_unet',type=int, default=32, help='n_channels for unet')
parser.add_argument('--n_channels_disc',type=int, default=32, help='n_channels for disc')
parser.add_argument('--use_cloob_unet',type=int, default=1, help='use_cloob for unet (0 for False)')
parser.add_argument('--use_cloob_disc',type=int, default=1, help='use_cloob for disc(0 for False)')
parser.add_argument('--n_cloob_channels_unet',type=int, default=256, help='n_cloob_channels for unet')
parser.add_argument('--n_cloob_channels_disc',type=int, default=256, help='n_cloob_channels for disc')
parser.add_argument('--n_time_channels',type=int, default=-1, help='time emb (-1 for n_channels*4)')
parser.add_argument('--denom_factor',type=int, default=16, help='for time emb. low default of 16.')

parser.add_argument('--n_steps',type=int, default=5, help='How many steps')
parser.add_argument('--beta_min',type=float, default=0.3, help='variance schedule')
parser.add_argument('--beta_max',type=float, default=0.9, help='variance schedule')

parser.add_argument('--weight_decay',type=float, default=1e-6, help='weight_decay')

parser.add_argument('--recon_loss_scale',type=float, default=1, help='How much weight do we put on recon loss')

parser.add_argument('--pct_text',type=float, default=0.1, help='What percentage text vs im for cloob embed. default 0.5')

args = parser.parse_args()
print('Training args:\n')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))

train(args)
print('Success!!')