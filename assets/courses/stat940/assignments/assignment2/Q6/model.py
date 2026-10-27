"""STAT 940 Assignment 2: complete the five TODOs for Q6(a)."""
import torch
from torch import nn

TAU = 0.25


def sample_latent(mu, logvar, generator):
    # TODO: implement sample_latent as specified in Q6(a).
    raise NotImplementedError("Complete sample_latent for Q6(a)")


def analytic_kl(mu, logvar):
    # TODO: implement analytic_kl as specified in Q6(a).
    raise NotImplementedError("Complete analytic_kl for Q6(a)")


class VAE(nn.Module):
    def __init__(self):
        # TODO: implement __init__ as specified in Q6(a).
        raise NotImplementedError("Complete __init__ for Q6(a)")

    def encode(self, X):
        # TODO: implement encode as specified in Q6(a).
        raise NotImplementedError("Complete encode for Q6(a)")

    def forward(self, X, generator):
        mu, logvar = self.encode(X)
        return self.decoder(sample_latent(mu, logvar, generator)), mu, logvar


def loss_terms(X, reconstructed, mu, logvar):
    # TODO: implement loss_terms as specified in Q6(a).
    raise NotImplementedError("Complete loss_terms for Q6(a)")
