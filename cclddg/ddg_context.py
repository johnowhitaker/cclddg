# AUTOGENERATED! DO NOT EDIT! File to edit: 01_DDG_Context.ipynb (unless otherwise specified).

__all__ = ['DDG_Context']

# Cell
import torch
from PIL import Image
import numpy as np

class DDG_Context():
    """TODO docstring"""
    def __init__(self, n_steps=5, beta_min=0.3, beta_max=0.9, device='cpu'):
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        """Gather consts for $t$ and reshape to feature map shape"""
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

    def q_xt_xtminus1(self, xtm1, t):
        """A single noising step:"""
        mean = self.gather(1. - self.beta, t) ** 0.5 * xtm1 # √(1−βt)*xtm1
        var = self.gather(self.beta, t) # βt I
        eps = torch.randn_like(xtm1) # Noise shaped like xtm1
        return mean + (var ** 0.5) * eps, eps

    def q_xt_x0(self, x0, t):
        """Jump to a given step"""
        mean = self.gather(self.alpha_bar, t) ** 0.5 * x0 # now alpha_bar
        var = 1-self.gather(self.alpha_bar, t) # (1-alpha_bar)
        eps = torch.randn_like(x0)
        return mean + (var ** 0.5) * eps, eps

    def p_xt(xt, noise, t):
        """The reverse step, not used in DDG"""
        alpha_t = self.gather(self.alpha, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
        mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise) # Note minus sign
        var = self.gather(self.beta, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    def tensor_to_image(self, t):
      return Image.fromarray(np.array(((t.detach().cpu().squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

    # Examples with some propmts
    def examples(self, ae_model, unet, cloob, n_examples=12, cfg_scale_max=4,
             prompts = [
                'A photograph portrait of a man with a beard, a human face',
                'Green hills and grass beneath a blue sky',
                'A watercolor painting of an underwater submarine',
                'A car, a photo of a red car',
                'An armchair in the shape of an avocado',
                'blue ocean waves',
                'A red stop sign'],
             img_size=128, z_dim=8,
            ):
        """Given ae_model, a u_net and cloob, produce some example images with CFG."""

        device = ae_model.device
        cfg_scale = torch.linspace(0, cfg_scale_max, n_examples).to(device)

        im_out = Image.new('RGB', (img_size*n_examples, img_size*len(prompts)))

        for i, p in enumerate(prompts):
            z = torch.randn((n_examples,z_dim), device=device)
            c = cloob.text_encoder(cloob.tokenize([p]*n_examples).to(device)).float()
            c_neg = torch.zeros((n_examples,512), device=device)
            x = torch.randn(n_examples, 4, img_size//8, img_size//8).to(device)
            t = torch.ones((n_examples,), dtype=torch.long).to(device)*self.n_steps
            while t[0] > 0:
                pred_im_pos = unet(x.float(), t, c, z)
                pred_im_neg = unet(x.float(), t, c_neg, z)
                pred_im = pred_im_neg + (pred_im_pos-pred_im_neg)*cfg_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 4, img_size//8, img_size//8)
                x, n = self.q_xt_x0(pred_im, t-1)
                t -= 1
                if t[0]==0:
                    for s in range(n_examples):
                        im_out.paste(self.tensor_to_image(ae_model.decode(pred_im[s].unsqueeze(0))), (img_size*s, img_size*i))

        return im_out
