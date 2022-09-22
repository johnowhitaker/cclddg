# Prepare:
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

# Assumes you're not in the cclddg directory so cp this to parent dir where the model above is 
# downloaded (along with the other github repos)

import sys
sys.path.append('cclddg')
sys.path.append('./latent-diffusion')
sys.path.append('./cloob-training')
from cclddg.core import UNet, Discriminator
from cclddg.ddg_context import DDG_Context
from cclddg.data import get_celebA_dl, get_cc12m_dl, get_imagewoof_dl, tensor_to_image
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
import os
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
    if args.dataset == 'celebA':
        data = get_celebA_dl(batch_size=args.batch_size, img_size=args.img_size)
    elif args.dataset == 'imagewoof': # TODO accept URL in case of local training
        data = get_imagewoof_dl(batch_size=args.batch_size, img_size=args.img_size)
    elif args.dataset == 'cc12m': # TODO accept URL in case of local training
        data = get_cc12m_dl(batch_size=args.batch_size, img_size=args.img_size)
    else:
        print('Dataset not recognized')
        return 0
    data_iter = iter(data)

    # Create the model
    unet = UNet(image_channels=4, n_channels=args.n_channels_unet,
               z_dim=args.z_dim, n_z_channels=args.n_z_channels,
                ch_mults=args.ch_mults, is_attn = args.is_attn
               ).to(device) # 4 or 8 or whatever based on ae
    disc = Discriminator(image_channels=4*2, n_channels=args.n_channels_disc,
                        ch_mults=args.ch_mults_disc, is_attn = args.is_attn_disc).to(device) # image_channels=4 but expects 8 since we condition on xt

    # Set up the DDG context
    ddg_context = DDG_Context(n_steps=args.n_steps, beta_min=args.beta_min, 
                              beta_max=args.beta_max, device=device)



    criterion = nn.BCELoss().to(device)

    losses = [] # Store losses for later plotting

    optim_gen = torch.optim.AdamW(unet.parameters(), lr=args.lr_gen, weight_decay=args.weight_decay) # Optimizer
    optim_dis = torch.optim.AdamW(disc.parameters(), lr=args.lr_disc, weight_decay=args.weight_decay) # Optimizer

    # Code for storing examples
    def log_examples(prompts_file=''):
        if prompts_file == '':
            if args.dataset == 'celebA':
                im = ddg_context.examples(ae_model, unet, cloob, n_examples=args.n_examples, 
                                          cfg_scale_min=args.cfg_min, cfg_scale_max=args.cfg_max,
                                 prompts=['A male face', 'a female face', 'a mugshot',
                                         'A photo of a face', 'A man with a beard'],
                                 img_size = args.img_size, z_dim=args.z_dim)
            else:
                im = ddg_context.examples(ae_model, unet, cloob, n_examples=args.n_examples,
                                          cfg_scale_min=args.cfg_min, cfg_scale_max=args.cfg_max,
                                 prompts=['A picture of a female face',
                                          'A watercolor painting of an underwater submarine',
                                          'A red STOP sign',
                                          'A landscape photo of green hills beneath a clear blue sky',
                                          'A group of people stand together chatting',
                                          'An armchair that is shaped like an avocado, product photo comfy avo chair',
                                          'Blue ocean waves'
                                         ],
                                 img_size = args.img_size, z_dim=args.z_dim)

            if args.wandb_project != 'None':
                wandb.log({'Examples':wandb.Image(im)})
            else:
                im.save('example.jpg')
                
        else: # We have a prompts file
            if args.wandb_project != 'None':
                with open(prompts_file, 'r') as pf:
                    prompts = pf.readlines()
                    table_data = []
                    for p in prompts:
                        eg_im = ddg_context.examples(ae_model, unet, cloob, n_examples=args.n_examples, 
                                                     cfg_scale_min=args.cfg_min, cfg_scale_max=args.cfg_max,
                                     prompts=[p], img_size = args.img_size, z_dim=args.z_dim) #
                        table_data.append([p, wandb.Image(eg_im)])
                    columns=['Prompt', f'Examples (cfg_scale from {args.cfg_min} to {args.cfg_max})'] # Rendered images as 'Image'
                    table = wandb.Table(columns=columns, data=table_data)
                    wandb.log({'Examples':table})
                    

    def log_reconstruction():
        # Get a batch of images and captions
        images, texts = next(iter(data))
        images = images.cuda()*2-1
        batch_size=images.shape[0]
        x0 = ae_model.encode(images).mode() # To latents (might want resize)
        t = torch.ones((batch_size,), dtype=torch.long).to(device)*(ddg_context.n_steps-1)
        z = torch.randn((batch_size,args.z_dim), device=device) # It's a gan now yay
        batch = cloob.normalize(scale_224(images.cuda()))
        cloob_embeds = cloob.image_encoder(batch).float()
        c = cloob_embeds # torch.zeros((batch_size,512), device=device) # Cloob embedding optional TODO zero out 10% 

        # Get the noised images (xt) and the noise (our target) plus the x(t-1)
        xt, eps_t = ddg_context.q_xt_x0(x0, t) # Most of the noise

        ims = []
        for i in range(5):
            ims.append(tensor_to_image(ae_model.decode(x0[i].unsqueeze(0))))

        while t[0] > 0:
            tm1 = torch.tensor(t.cpu().numpy()-1, dtype=torch.long).to(device)
            with torch.no_grad():
                pred_im = unet(xt.float(), t, c, z)
            for i in range(5):
                ims.append(tensor_to_image(ae_model.decode(xt[i].unsqueeze(0)))) # Change to x to see noised version as in old one
            for i in range(5):
                ims.append(tensor_to_image(ae_model.decode(pred_im[i].unsqueeze(0)))) # Change to x to see noised version as in old one
            xt, n = ddg_context.q_xt_x0(x0, tm1)
            t -= 1

        im = Image.new('RGB', (ims[0].size[0]*5, ims[0].size[1]*(ddg_context.n_steps*2-1)))
        for i, img in enumerate(ims):
            im.paste(img, ((i%5)*img.size[0], (i//5)*img.size[1]))
        if args.wandb_project != 'None':
            wandb.log({'Reconstruction':wandb.Image(im)})

    for i in tqdm(range(0, args.n_batches)): # Run through the dataset

        # Get a batch of images and captions
        try:
            images, texts = next(data_iter)
        except StopIteration:
            data_iter = iter(data) # Restart 
            images, texts = next(data_iter)
            
        images = images.cuda()*2-1
        batch_size=images.shape[0]

        log = {} # Prepare logging

        unet.zero_grad() # Zero the gradients
        disc.zero_grad() # Zero the gradients

        # Prepare data
        x0 = ae_model.encode(images).mode() # To latents (might want resize)
        t = torch.randint(1, ddg_context.n_steps, (batch_size,), dtype=torch.long).to(device) # Random 't's 
        z = torch.randn((batch_size,args.z_dim), device=device) # It's a gan now yay
        
        # Get cloob embeddings TODO check if pct_text is 0 or 1 to save time
        if args.use_cloob_unet or args.use_cloob_disc:
            with torch.no_grad():
                batch = cloob.normalize(scale_224(images.cuda()))
                cloob_embeds_images = cloob.image_encoder(batch).float()
                cloob_embeds_texts =  cloob.text_encoder(cloob.tokenize(texts, truncate=True).to(device)).float()
                mask = (torch.cuda.FloatTensor(batch_size, 1).uniform_() < args.pct_text).float().expand(-1, 512)
                c = mask*cloob_embeds_texts + (1-mask)*cloob_embeds_images
        else:
            c =  torch.zeros((batch_size,512), device=device) # if not using cloob
        
        # zero out 10%: TODO set with arg
        zero_mask = (torch.cuda.FloatTensor(batch_size, 1).uniform_() < 1-args.pct_zeros).float().expand(-1, 512)
        c = c*zero_mask

        # Get the noised images (xt) and the noise (our target) plus the x(t-1)
        xtm1, eps_t = ddg_context.q_xt_x0(x0, t-1) # Most of the noise
        xt, eps_added = ddg_context.q_xt_xtminus1(xtm1, t) # One extra step
        xt.requires_grad = True # Only needed if doing R1 reg

        # Disc loss on the 'real' samples
        disc_input = torch.cat((xt, xtm1), dim=1)
        disc_pred_real = disc(disc_input, t, c) # Predict for xtm1 conditioned on xt
        label = torch.ones_like(disc_pred_real) 
        disc_loss_real = criterion(disc_pred_real, label)
        disc_loss_real.backward(retain_graph=True) # Only needed if doing R1 reg
        log['disc_loss_real'] = disc_loss_real.item()
        log['D(real).mean()'] = disc_pred_real.mean().item()
        
        # R1 regularization term
        # NB This bit I copied from nvlabs code, first time I've used that even as ref. Not sure how this affects licence. 
        r1_gamma=1 # TODO arg
        grad_real = torch.autograd.grad(outputs=disc_pred_real.sum(), inputs=xt, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = r1_gamma / 2 * grad_penalty
        grad_penalty.backward()

        # Disc on a fake batch
        gen_pred_x0 = unet(xt.float(), t, c, z) # Run xt through the network to get its predictions
        gen_pred_noised_to_xtm1, noise_added_to_pred = ddg_context.q_xt_x0(gen_pred_x0, t-1)
        label.fill_(0) # Change labels to 0
        disc_pred_fake = disc(torch.cat((xt, gen_pred_noised_to_xtm1.detach()), dim=1), t, c) # detach to avoid going through graph twice
        disc_loss_fake= criterion(disc_pred_fake, label)
        disc_loss_fake.backward()
        log['disc_loss_fake'] = disc_loss_fake.item()
        log['D(fake).mean()'] = disc_pred_fake.mean().item()
        log['disc_loss_sum'] = disc_loss_fake.item() +  disc_loss_real.item()
        optim_dis.step() # Update the discriminator 
        disc.zero_grad() # Zero out grads again (can also put in eval mode)

        ## TRAIN THE GENERATOR PART
        # Disc loss:
        unet.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        disc_pred_fake_G = disc(torch.cat((xt, gen_pred_noised_to_xtm1), dim=1), t, c)
        disc_loss_fake_G = criterion(disc_pred_fake_G, label)
        disc_loss_fake_G.backward(retain_graph=True) # We may also want to update based on recon loss
        log['disc_loss_fake_G'] = disc_loss_fake_G.item()
        log['D(fake).mean() G'] = disc_pred_fake_G.mean().item()

        # Reconstruction loss. Set recon_loss_scale=0 to skip (will still be recorded)
        l = F.mse_loss(x0.float(), gen_pred_x0) # Compare the predictions with the targets
        log['recon_loss'] = l.item()
        if args.recon_loss_scale > 0:
            recon_loss = args.recon_loss_scale*l
            recon_loss.backward()

        optim_gen.step() # Update the discriminator 

        losses.append(log)
        if args.wandb_project != 'None':
            wandb.log(log)

        # Occasionally log demo images
        if (len(losses))%args.log_images_every==0:
            log_examples(args.prompts_file)
            log_reconstruction()
            
        # Occasionally save models
        if (len(losses))%args.save_models_every==0:
            torch.save(unet.state_dict(), f'unet_cc12m_{len(losses):06}.ckpt')
            torch.save(disc.state_dict(), f'disc_cc12m_{len(losses):06}.ckpt')
            
        # TODO add a CLOOB metric for text-to-image runs

    if args.wandb_project != 'None':
        wandb.finish()
    
import argparse
parser = argparse.ArgumentParser(description='Train CCLDDG (quick test script)')
parser.add_argument('--n_batches',type=int, default=10, help='How many batches should we train on')
parser.add_argument('--dataset',type=str, default='celebA', help='What dataset?')
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

# TODO help for these
parser.add_argument('--ch_mults', nargs='+', type=int, default=(1, 2, 2, 4))
parser.add_argument('--is_attn', nargs='+', type=int, default=(0, 0, 1, 1))
parser.add_argument('--ch_mults_disc', nargs='+', type=int, default=(1, 2, 2, 4))
parser.add_argument('--is_attn_disc', nargs='+', type=int, default=(0, 0, 1, 1))

parser.add_argument('--n_steps',type=int, default=5, help='How many steps')
parser.add_argument('--beta_min',type=float, default=0.3, help='variance schedule')
parser.add_argument('--beta_max',type=float, default=0.9, help='variance schedule')

parser.add_argument('--weight_decay',type=float, default=1e-6, help='weight_decay')

parser.add_argument('--recon_loss_scale',type=float, default=1, help='How much weight do we put on recon loss')

parser.add_argument('--pct_text',type=float, default=0.1, help='What percentage text vs im for cloob embed. default 0.1')

parser.add_argument('--pct_zeros',type=float, default=0.1, help='What percentage should we zero out CLOOB embeddings for CGF, default 0.1 (10%)')

parser.add_argument('--prompts_file', type=str, default='', help='Prompts (one per line)')

parser.add_argument('--cfg_min',type=float, default=0, help='min CFG scale for example ims (default 0)')
parser.add_argument('--cfg_max',type=float, default=2, help='max CFG scale for example ims (default 2)')
parser.add_argument('--n_examples',type=float, default=7, help='examples_per_prompt')


args = parser.parse_args()
print('Training args:\n')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))

train(args)
print('Success!!')