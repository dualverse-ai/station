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

import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from ray import tune
from typing import Dict, Any, Optional

class BottleneckDilatedBlock(nn.Module):
    features: int; bottleneck_ratio: float = 0.28; dilation: int = 6
    @nn.compact
    def __call__(self, x):
        mid = max(1, int(self.features * self.bottleneck_ratio))
        y = nn.Conv(mid,(1,1),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(x); y = nn.relu(y)
        y = nn.Conv(mid,(3,3),padding='SAME',kernel_dilation=(self.dilation,self.dilation),
                    kernel_init=nn.initializers.xavier_uniform())(y); y = nn.relu(y)
        y = nn.Conv(self.features,(1,1),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(y)
        return nn.relu(x + y)

class ConvLSTMCellLN(nn.Module):
    features: int
    @nn.compact
    def __call__(self, carry, x):
        h, c = carry
        gates = nn.Conv(4*self.features,(3,3),padding='SAME',
                        kernel_init=nn.initializers.xavier_uniform())(jnp.concatenate([x,h],axis=-1))
        gates = nn.LayerNorm(epsilon=1e-5)(gates)
        i,f,g,o = jnp.split(gates, 4, axis=-1)
        i = nn.sigmoid(i); f = nn.sigmoid(f); o = nn.sigmoid(o); g = jnp.tanh(g)
        c = f*c + i*g
        h = o*jnp.tanh(c)
        return (h,c), h

class NetRIN_AttnGap(nn.Module):
    rin_alpha: float = 0.33
    cnn_features_1: int = 32; cnn_features_2: int = 64
    convlstm_features: int = 64; dilation: int = 6; bottleneck_ratio: float = 0.28
    head_hidden: int = 48; steps: int = 4
    @nn.compact
    def __call__(self, x: jnp.ndarray, done: jnp.ndarray, rnn_state: Optional[Dict]=None):
        B = x.shape[0]
        x = nn.Conv(self.cnn_features_1,(3,3),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(x); x = nn.relu(x)
        x = BottleneckDilatedBlock(self.cnn_features_1, self.bottleneck_ratio, self.dilation)(x)
        x = nn.Conv(self.cnn_features_2,(3,3),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(x); x = nn.relu(x)
        x = BottleneckDilatedBlock(self.cnn_features_2, self.bottleneck_ratio, self.dilation)(x)
        x_ln = nn.LayerNorm(epsilon=1e-5)(x)
        x = x + self.rin_alpha * (x_ln - x)
        if rnn_state is None:
            zeros = lambda: jnp.zeros((B,8,8,self.convlstm_features), dtype=x.dtype)
            rnn_state = {'h': zeros(), 'c': zeros()}
        if done is not None:
            m = (1.0 - done.reshape(B,1,1,1).astype(x.dtype))
            rnn_state = {'h': rnn_state['h']*m, 'c': rnn_state['c']*m}
        h, c = rnn_state['h'], rnn_state['c']
        for _ in range(self.steps):
            (h,c), _ = ConvLSTMCellLN(self.convlstm_features)((h,c), x)
        B_,H,W,C = h.shape
        attn_logits = nn.Conv(1,(1,1),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(h).reshape((B_,H*W))
        attn_w = nn.softmax(attn_logits, axis=-1).reshape((B_,H*W,1))
        tokens = h.reshape((B_,H*W,C))
        attn_vec = (attn_w * tokens).sum(axis=1)
        gap_vec = tokens.mean(axis=1)
        z = jnp.concatenate([attn_vec, gap_vec], axis=-1)
        shared = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())(z); shared = nn.relu(shared)
        ph = nn.Dense(self.head_hidden, kernel_init=nn.initializers.xavier_uniform())(shared); ph = nn.relu(ph)
        policy = nn.Dense(4, kernel_init=nn.initializers.xavier_uniform())(ph)
        vh = nn.Dense(self.head_hidden, kernel_init=nn.initializers.xavier_uniform())(shared); vh = nn.relu(vh)
        value = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(vh).squeeze(-1)
        return policy, value, {'h': h, 'c': c}

BASE_SEED = 42

def _define_hyperparameters():
    return {
        'learning_rate': tune.choice([3.5e-4]),
        'entropy_coef': tune.choice([0.009, 0.010, 0.011]),
        'value_loss_coef': tune.choice([0.55]),
        'cnn_features_1': tune.choice([32]),
        'cnn_features_2': tune.choice([64]),
        'convlstm_features': tune.choice([64]),
        'dilation': tune.choice([6]),
        'bottleneck_ratio': tune.choice([0.28]),
        'head_hidden': tune.choice([48]),
        'steps': tune.choice([4]),
    }

def create_network(hparams: Dict[str, Any]):
    return NetRIN_AttnGap(
        rin_alpha=0.33,
        cnn_features_1=int(hparams['cnn_features_1']),
        cnn_features_2=int(hparams['cnn_features_2']),
        convlstm_features=int(hparams['convlstm_features']),
        dilation=int(hparams['dilation']),
        bottleneck_ratio=float(hparams['bottleneck_ratio']),
        head_hidden=int(hparams['head_hidden']),
        steps=int(hparams['steps']),
    )

def create_optimizer(learning_rate: float = 3.5e-4):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))