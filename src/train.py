import os
import numpy as np

import GPUtil
from rich import print
from rich.console import Console
from rich.progress import track
import torch
import torch.distributions as td
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb
import names

from .loader import Omniglot
from .model import VAE
from .shape import Shape

if __name__ == "__main__":

    log = True
    # log = False

    if log:
        wandb.init(project='simple-vae')
        name = wandb.run.name
    else:
        name = names.get_full_name().replace(' ', '-').lower()
    results_dir = 'results'
    save_dir = os.path.join(results_dir, name)

    console = Console()

    device = 'cuda'
    batch_size = 16

    # dataset = Omniglot('datasets/omniglot')
    dataset = Omniglot('datasets/omniglot/images_background/Greek')
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    hid_size, hid_chan = 200, 64
    image_shape = Shape(batch=batch_size, chan=1,
                        height=dataset.size, width=dataset.size)

    n_layers = 5
    GPUtil.showUtilization()
    vae = VAE(image_shape, hid_size, hid_chan).to(device)
    GPUtil.showUtilization()

    # run = 'wobbly-wave-44'
    run = None
    if run is not None:
        console.log(f'Loading Run {run}')
        load_dir = os.path.join(results_dir, run, 'model.pth')
        state_dict = torch.load(load_dir)
        vae.load_state_dict(state_dict)

    optimizer = Adam(vae.parameters(), lr=2e-5)

    def make_image(x: torch.Tensor):
        x = x[0]
        min, max = x.min(), x.max()
        x = (x.clamp(0, 1) * 255).to(torch.uint8)
        x = x.detach().cpu().numpy()
        return x, min, max

    for e in range(5000):

        try:
            for i, x in track(enumerate(dataloader),
                              description=f'Training Epoch {e:5}',
                              total=len(dataloader)):

                Q, P, z = vae(x.to(device))

                # Construction prior distribution
                prior = td.Normal(torch.zeros_like(z), torch.ones_like(z))
                prior = td.Independent(prior, 1)

                # KL divergence term
                div = Q.log_prob(z) - prior.log_prob(z)

                # Reconstruction term
                rec = - P.log_prob(x.to(device))

                # ELBO loss
                elbo = (rec + div).mean()

                elbo.backward()
                nn.utils.clip_grad.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer.step()

                if log:
                    wandb.log({'divergence': div.mean().item(),
                               'reconstruction': rec.mean().item(),
                               'loss': elbo.item()})
                if i % 50 == 0:
                    print((f'Step {i:<4} | ELBO: {elbo.item():<10.5f} '
                           f'div: {div.mean().item():<10.5f} rec: {rec.mean().item():<10.5f}'))

            if log and e % 50 == 0:
                print('Generating Images')
                x_image = x[0].detach().cpu().numpy()
                x_min, x_max = np.min(x_image), np.max(x_image)
                m_image, m_min, m_max = make_image(P.mean)
                sample = vae.p(prior.sample()).mean
                s_image, s_min, s_max = make_image(sample)
                wandb.log({'Input Image': wandb.Image(x_image),
                           'Mean Image': wandb.Image(m_image),
                           'Sampled Image': wandb.Image(s_image)})
                wandb.log({'Input Image Max': x_max,
                           'Input Image Min': x_min,
                           'Mean Image Max': m_max,
                           'Mean Image Min': m_min,
                           'Sampled Image Max': s_max,
                           'Sampled Image Min': s_min})

                print('Saving Model')
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                state_dict = vae.state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

        except KeyboardInterrupt:
            print('Shutting Down')
            if log:
                print('Saving Model')
                wandb.finish(quiet=True)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                state_dict = vae.state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'model.pth'))
            break
