#!/usr/bin/env python3
# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
import optax

def mae_with_temporal_curvature(predictions, targets, params, x, lam=1e-4):
    mae = jnp.mean(jnp.abs(predictions - targets))
    curv = predictions[:, 2:, :] - 2.0 * predictions[:, 1:-1, :] + predictions[:, :-2, :]
    curv_pen = jnp.mean(curv * curv)
    return mae + lam * curv_pen

class HyperGain(nn.Module):
    output_dim: int
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(32)(x); h = nn.relu(h)
        # Initialize gain to be close to 1
        return nn.Dense(self.output_dim, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal'))(h) + 1.0

class HyperResidualCopyHead(nn.Module):
    num_neurons: int
    embedding_dim: int
    drop: float = 0.05
    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T_in, N = x.shape
        x_flat = x.transpose(0, 2, 1).reshape(B * N, T_in)

        neuron_indices = jnp.arange(self.num_neurons)
        neuron_embeddings = nn.Embed(num_embeddings=self.num_neurons, features=self.embedding_dim)(neuron_indices)
        tiled_embeddings = jnp.tile(neuron_embeddings, (B, 1))

        gain1 = HyperGain(64, name="hyper_gain_residual")(tiled_embeddings)

        h = nn.Dense(64)(x_flat) * gain1 # Multiplicative conditioning
        h = nn.relu(h)
        h = nn.Dropout(rate=self.drop)(h, deterministic=not training)
        h = nn.Dense(32)(h)
        residual = h.reshape(B, N, 32).transpose(0, 2, 1)

        out = jnp.zeros((B, 32, N), dtype=residual.dtype)
        out = out.at[:, :T_in, :].set(x)
        return out + residual

class FourierForecasterLN_Ramp(nn.Module):
    rank_k: int; proj_rank: int; hidden_size: int; drop: float; num_neurons: int; embedding_dim: int;
    output_horizon: int = 32; learn_gamma: bool = True
    @nn.compact
    def __call__(self, x, training: bool = False):
        # ... (SOTA Fourier branch code is identical and omitted for brevity) ...
        B, T_in, N = x.shape; k, p = self.rank_k, self.proj_rank
        U = self.param('U', nn.initializers.lecun_normal(), (N, p)); V = self.param('V', nn.initializers.lecun_normal(), (k, p))
        U_eff = U / (jnp.linalg.norm(U, axis=0, keepdims=True) + 1e-6)
        x_fft = jnp.fft.rfft(x, axis=1); x_fft_bn = jnp.transpose(x_fft, (0, 2, 1))
        f_in_fft = jnp.einsum('bnf,np->bfp', x_fft_bn, U_eff) @ V.T
        f_ri = jnp.concatenate([f_in_fft.real, f_in_fft.imag], axis=-1); f_flat = f_ri.reshape(B, -1)
        h = nn.Dense(self.hidden_size)(f_flat); h = nn.relu(h); h = nn.Dropout(rate=self.drop)(h, deterministic=not training)
        h = nn.Dense(self.hidden_size)(h); h = nn.relu(h); h = nn.Dropout(rate=self.drop)(h, deterministic=not training)
        F_out = self.output_horizon // 2 + 1; out_flat = nn.Dense(F_out * k * 2)(h)
        out_ri = out_flat.reshape(B, F_out, k, 2); f_fft = out_ri[..., 0] + 1j * out_ri[..., 1]
        f_fft_n = nn.LayerNorm()(f_fft.real) + 1j * nn.LayerNorm()(f_fft.imag)
        y_fft = (f_fft_n @ V) @ U_eff.T; y_fourier = jnp.fft.irfft(y_fft, n=self.output_horizon, axis=1)
        gamma_raw = self.param('gamma_raw', nn.initializers.zeros, (self.output_horizon,)); gamma = nn.sigmoid(gamma_raw)[None, :, None]
        y_fourier = gamma * y_fourier
        y_copy = HyperResidualCopyHead(num_neurons=self.num_neurons, embedding_dim=self.embedding_dim, drop=self.drop)(x, training=training)
        return y_copy + y_fourier

BASE_SEED = 42
BATCH_SIZE = 8

def _define_hyperparameters():
    return {
        'learning_rate': 9e-4, 'rank_k': 320, 'proj_rank': 36, 'hidden_size': 160,
        'drop': 0.05, 'learn_gamma': True, 'embedding_dim': 8, 'num_neurons': 71721
    }

class Wrapper:
    def __init__(self, hparams):
        model_params = {k:v for k,v in hparams.items() if k not in ['learning_rate']}
        self.model = FourierForecasterLN_Ramp(**model_params)
        self.needs_rng = True; self.mutable = []
    def init(self, rng_key, dummy_input):
        rng_params, rng_dropout = random.split(rng_key)
        return self.model.init({'params': rng_params, 'dropout': rng_dropout}, dummy_input, training=True)
    def apply(self, params, x, training=False, mutable=None, rngs=None):
        return self.model.apply(params, x, training=training, rngs=(rngs or {}))

def create_network(hparams): return Wrapper(hparams)
def compute_loss(p, t, pa, x): return mae_with_temporal_curvature(p, t, pa, x, lam=1e-4)
def create_optimizer(lr):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))