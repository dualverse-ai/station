## Hypothesis Test 2: Reducing Orthogonality Regularization
## d478cf9c-3d67-48a3-a9da-afaa7874201b

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Sequence, Dict, Any

BATCH_SIZE = 20

def _define_hyperparameters():
    return {
        'learning_rate': 1e-3, 'rank_k': 128, 'hidden_h': 24,
        'use_factor_drop': True, 'factor_dropout_rate': 0.06, 'use_residuals': True,
        'transformer_heads': 4, 'transformer_layers': 2,
        'num_encoder_experts': 4, 
        'lambda_ortho': 1e-8  # REDUCED from 1e-6
    }

class LearnablePositionalEncoding(nn.Module):
    num_features: int; seq_len: int
    @nn.compact
    def __call__(self, x):
        position_indices = jnp.arange(0, self.seq_len, dtype=jnp.int32)
        pos_embeddings_table = nn.Embed(num_embeddings=self.seq_len, features=self.num_features, name="learnable_pos_embeddings")
        learned_pe = pos_embeddings_table(position_indices)
        return x + learned_pe[None, :, :]

class GatingCNN(nn.Module):
    num_experts: int
    @nn.compact
    def __call__(self, x):
        x_transposed = jnp.transpose(x, (0, 2, 1))
        gate_features = nn.Conv(features=16, kernel_size=(3,), padding='SAME')(x_transposed)
        gate_features = nn.relu(gate_features)
        gate_features_flat = gate_features.reshape(gate_features.shape[0], gate_features.shape[1], -1)
        gating_logits = nn.Dense(features=self.num_experts)(gate_features_flat)
        return gating_logits

class MoE2_PFFT_Ortho(nn.Module):
    hparams: Dict
    num_neurons: int = 71721
    @nn.compact
    def __call__(self, x, training: bool = False):
        K, H, C = self.hparams['rank_k'], self.hparams['hidden_h'], self.hparams['num_encoder_experts']
        
        gating_logits = GatingCNN(num_experts=C, name="EncoderGatingCNN")(x)
        gating_weights = nn.softmax(gating_logits, axis=-1)
        expert_encoders = self.param('expert_encoders', nn.initializers.lecun_normal(), (C, self.num_neurons, K))
        W_in_eff = jnp.einsum('bnc,cnk->bnk', gating_weights, expert_encoders)
        z = jnp.einsum('btn,bnk->btk', x, W_in_eff)
        
        z_pos = LearnablePositionalEncoding(num_features=K, seq_len=x.shape[1])(z)
        z_fk = jnp.transpose(z_pos, (0, 2, 1))
        W1 = self.param('W1', nn.initializers.lecun_normal(), (K, x.shape[1], H))
        W2 = self.param('W2', nn.initializers.lecun_normal(), (K, H, 32))
        h = jnp.einsum('bkt,kth->bkh', z_fk, W1); h = nn.tanh(h)
        z_future = jnp.einsum('bkh,kho->bko', h, W2); z_future = jnp.transpose(z_future, (0, 2, 1))

        if self.hparams['use_factor_drop'] and training:
            keep_prob = 1.0 - self.hparams['factor_dropout_rate']; rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, shape=(1, 1, K)).astype(z_future.dtype)
            z_future = z_future * (mask / keep_prob)
        
        z_interaction = z_future
        for i in range(self.hparams['transformer_layers']):
            norm_z = nn.LayerNorm()(z_interaction)
            attn_output = nn.SelfAttention(num_heads=self.hparams['transformer_heads'], name=f"self_attention_{i}")(norm_z)
            z_interaction = z_interaction + attn_output
        z_refined = z_interaction
        
        expert_decoders = self.param('expert_decoders', nn.initializers.lecun_normal(), (C, self.num_neurons, K))
        W_out_eff = jnp.einsum('bnc,cnk->bnk', gating_weights, expert_decoders)
        y = jnp.einsum('btk,bnk->btn', z_refined, W_out_eff)
        
        if self.hparams['use_residuals']:
            alpha = self.param('alpha', nn.initializers.zeros, (32,)); y = y + alpha[None, :, None] * x[:, -1, :][:, None, :]
            beta = self.param('beta', nn.initializers.zeros, (32,)); velocity = x[:, -1, :] - x[:, -2, :]; y = y + beta[None, :, None] * velocity[:, None, :]
        return y

class ModelWrapper:
    needs_rng = True
    def __init__(self, hparams: Dict): self.hparams = hparams; self.model = MoE2_PFFT_Ortho(hparams=hparams)
    def init(self, rng_key, dummy_input): return self.model.init({'params': rng_key, 'dropout': rng_key}, dummy_input, training=True)
    def apply(self, params, x, training=False, rngs=None): return self.model.apply(params, x, training=training, rngs={'dropout': rngs.get('dropout')} if rngs else None)

def create_network(hparams: Dict[str, Any]): return ModelWrapper(hparams)
def create_optimizer(learning_rate: float = 1e-3): return optax.adamw(learning_rate=learning_rate, weight_decay=5e-7, eps=1e-8)
def compute_loss(predictions, targets, variables, x):
    params = variables['params']
    hparams = _define_hyperparameters()
    lambda_ortho = hparams['lambda_ortho']
    
    mae = jnp.mean(jnp.abs(predictions - targets))
    d2 = predictions[:, 2:, :] - 2.0 * predictions[:, 1:-1, :] + predictions[:, :-2, :]
    smooth = jnp.mean(jnp.square(d2))
    
    expert_encoders = params['expert_encoders']
    C, N, K = expert_encoders.shape
    
    def single_ortho_loss(W):
        gram = (W.T @ W) / (N + 1e-8)
        return jnp.mean(jnp.square(gram - jnp.eye(K, dtype=W.dtype)))
    
    ortho_losses = jax.vmap(single_ortho_loss)(expert_encoders)
    mean_ortho_loss = jnp.mean(ortho_losses)

    return mae + 5e-5 * smooth + lambda_ortho * mean_ortho_loss
