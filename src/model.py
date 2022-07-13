import torch
import torch.nn as nn
import torch.distributions as td

from .shape import Shape


class Reshape (nn.Module):

    def __init__(self, shape: Shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Encoder (nn.Module):

    def __init__(self,
                 in_shape: Shape,
                 hid_size: int,
                 hid_chan: int,
                 n_conv: int = 3) -> None:

        super().__init__()
        self.hid_size = hid_size
        layers = []

        # Conv Layers
        conv_shape = in_shape
        conv_shape = conv_shape.conv(kernel_size=3, stride=1)
        layers += [nn.Conv2d(in_shape.get('chan'), hid_chan,
                             kernel_size=3, stride=1)]
        layers += [nn.ReLU()]
        for _ in range(n_conv):
            conv_shape = conv_shape.conv(out_chan=hid_chan,
                                         kernel_size=3, stride=1)
            layers += [nn.Conv2d(hid_chan, hid_chan,
                                 kernel_size=3, stride=1)]
            layers += [nn.ReLU()]

        # Linear Layers
        layers += [nn.Flatten()]
        layers += [nn.Linear(conv_shape.flat_size(), hid_size * 2)]

        layers += [nn.ReLU()]
        # layers += [nn.Softmax()]

        self.out_shape = conv_shape
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        mean, std = self.model(x).split(self.hid_size, dim=-1)
        Q = td.Normal(mean, (std / 2).exp())
        Q = td.Independent(Q, 1)
        return Q


class Decoder (nn.Module):

    def __init__(self,
                 out_shape: Shape,
                 hid_size: int,
                 hid_chan: int,
                 n_conv: int = 3) -> None:

        super().__init__()
        layers = []

        # Deconv Layers
        out_chan = out_shape.get('chan')
        conv_shape = out_shape
        for _ in range(n_conv):
            conv_shape = conv_shape.conv(out_chan=hid_chan,
                                         kernel_size=3, stride=1)
            layers += [nn.ConvTranspose2d(hid_chan, hid_chan,
                                          kernel_size=3, stride=1)]
            layers += [nn.ReLU()]
        conv_shape = conv_shape.conv(out_chan=out_chan,
                                     kernel_size=3, stride=1)
        layers += [nn.ConvTranspose2d(hid_chan, out_chan,
                                      kernel_size=3, stride=1)]
        layers += [nn.ReLU()]

        # Linear Layers
        in_shape = conv_shape.set('chan', hid_chan)
        layers = [nn.Linear(hid_size, in_shape.flat_size()),
                  Reshape(in_shape)] + layers

        layers += [nn.ReLU()]
        # layers += [nn.Softmax()]

        self.log_scale = nn.parameter.Parameter(torch.Tensor([0.0]))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        mean = self.model(x)
        P = td.Normal(mean, self.log_scale.exp())
        P = td.Independent(P, 3)
        return P


class VAE (nn.Module):

    def __init__(self,
                 in_shape: Shape,
                 hid_size: int,
                 hid_chan: int,
                 n_conv: int = 3) -> None:

        super().__init__()
        self.image_shape, self.hid_size = in_shape, hid_size
        self.q = Encoder(in_shape, hid_size, hid_chan, n_conv)
        self.p = Decoder(in_shape, hid_size, hid_chan, n_conv)

    def forward(self, x: torch.Tensor):
        Q = self.q(x)
        z = Q.sample()
        P = self.p(z)
        return Q, P, z


if __name__ == "__main__":

    hid_size, hid_chan = 200, 16
    out_shape = Shape(batch=5, chan=1, height=105, width=105)

    x = torch.rand(*out_shape)

    vae = VAE(out_shape, hid_size, hid_chan)

    Q, P, z = vae(x)

    print(x.shape, Q.sample().shape, P.sample().shape)
    print(Q.mean.shape, P.mean.shape)
