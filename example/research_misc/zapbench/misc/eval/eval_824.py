import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
import optax
import numpy as np 
import os
import sys
from flax.core import unfreeze, freeze
from flax import traverse_util

# Constants for optimizer (from Horizon I's code - reused here for consistency)
CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4

# ModelInterface class (modified apply to return full tuple for this test)
class ModelInterface:
    def __init__(self, model_class, hparams):
        self.model = model_class(num_neurons=hparams['num_neurons'], **{k: v for k,v in hparams.items() if k != 'num_neurons'})
    def init(self, rng_key, dummy_input): return self.model.init(rng_key, dummy_input)
    def apply(self, params, x, training=False):
        output = self.model.apply(params, x)
        return output

# --- Logos II's SOTA Architecture (ID 528) - Modified ---
class Attention(nn.Module):
  @nn.compact
  def __call__(self, query, keys, values):
    scores = jnp.einsum('bphd,bptd->bpht', query, keys) / jnp.sqrt(query.shape[-1])
    attn_weights = nn.softmax(scores, axis=-1)
    context = jnp.einsum('bpht,bptd->bphd', attn_weights, values)
    return context

class LD_PFR_Module(nn.Module):
  num_neurons: int; P: int; latent_hidden_dim: int; t_out: int = 32
  @nn.compact
  def __call__(self, x, V, A):
      batch_size = x.shape[0]
      s = jnp.einsum('btn,np->btp', x, V)
      gru_cell = nn.GRUCell(features=self.latent_hidden_dim)
      vmapped_gru = nn.vmap(nn.RNN, in_axes=2, out_axes=1, variable_axes={'params': None}, split_rngs={'params': False})
      gru_hidden_states = vmapped_gru(gru_cell)(s[:,:,:,None])
      output_queries = self.param('output_queries', nn.initializers.glorot_uniform(), (self.t_out, self.latent_hidden_dim))
      queries_bphd = jnp.tile(output_queries[None, None, :, :], (s.shape[0], self.P, 1, 1))
      context_vectors = Attention(name="Attention_Module")(queries_bphd, gru_hidden_states, gru_hidden_states)
      final_hidden_state = gru_hidden_states[:, :, -1, :]
      final_hidden_state_tiled = jnp.tile(final_hidden_state[:, :, None, :], (1, 1, self.t_out, 1))
      fused_representation = jnp.concatenate([context_vectors, final_hidden_state_tiled], axis=-1)
      fused_flat = fused_representation.reshape(batch_size, self.P, -1)
      h_prime = nn.Dense(features=self.t_out)(fused_flat)
      return jnp.einsum('bph,np->bhn', h_prime, A), s # Return s as intermediate for dynamic modulation

# MODIFIED CARG_Ablation_Module with DYNAMIC S_FLAT dependent modulation (SIGMOID ACTIVATION FIX)
class CARG_Ablation_Module(nn.Module):
  num_neurons: int; rank: int; num_clusters: int; corrector_hidden_dim: int; t_out: int = 32 
  @nn.compact
  def __call__(self, c, s_flat_for_modulation): # Now takes s_flat for dynamic modulation
      mod_factors = []
      mod_raw_outputs = [] # Store raw outputs before activation
      for i in range(self.num_clusters):
          # FIX: Explicitly set bias_init to 0.0 for sigmoid, and use sigmoid activation
          mod_net = nn.Dense(features=1, name=f"Modulation_{i}", bias_init=nn.initializers.constant(0.0)) 
          mod_raw_output = mod_net(s_flat_for_modulation) 
          mod_factors.append(0.5 + 1.0 * nn.sigmoid(mod_raw_output)) # FIX: Sigmoid activation, scaled to (0.5, 1.5)
          mod_raw_outputs.append(mod_raw_output)

      corrections_per_expert = []
      for i in range(self.num_clusters):
          h1 = nn.Dense(features=self.corrector_hidden_dim, name=f"Corrector_{i}_Dense1")(c)
          h1 = nn.relu(h1)
          
          h1_modulated = h1 * mod_factors[i][:, None, :] 
          
          output = nn.Dense(features=self.t_out, name=f"Corrector_{i}_Dense2")(h1_modulated)
          corrections_per_expert.append(output)
      
      # Stack individual corrector outputs before averaging, for complementarity analysis
      stacked_corrector_outputs = jnp.stack(corrections_per_expert, axis=0) # Shape (C, B, N, T_out)
      carg_residual_bnk = jnp.mean(stacked_corrector_outputs, axis=0) # Average for final residual
      return jnp.transpose(carg_residual_bnk, (0, 2, 1)), jnp.stack(mod_factors, axis=0), jnp.stack(mod_raw_outputs, axis=0), stacked_corrector_outputs # Return stacked individual outputs

# LD_Net_CARG_Ablation main network - Modified to integrate NECG and return intermediates
class LD_Net_CARG_Ablation(nn.Module):
  num_neurons: int; rank: int; Qn: int; P: int; latent_hidden_dim: int
  corrector_hidden_dim: int 
  num_clusters: int 
  @nn.compact
  def __call__(self, x):
      P_nfm = self.param('P_nfm', nn.initializers.normal(1/jnp.sqrt(4)), (self.Qn, 4, self.rank))
      E = self.param('E', nn.initializers.normal(1/jnp.sqrt(self.num_neurons)), (self.num_neurons, self.Qn))
      b_in = self.param('b_in', nn.initializers.zeros, (self.num_neurons, self.rank))
      B = self.param('B', nn.initializers.normal(1/jnp.sqrt(self.rank)), (self.rank, 32))
      x_bn4 = jnp.transpose(x, (0, 2, 1)); u = jnp.einsum('bnf,qfm->bnqm', x_bn4, P_nfm)
      c = jnp.einsum('bnqm,nq->bnm', u, E) + b_in # NFM latent features 'c'
      y_nfm = jnp.transpose(jnp.einsum('bnm,mk->bnk', c, B), (0, 2, 1))
      V = self.param('V', nn.initializers.normal(1/jnp.sqrt(self.num_neurons)), (self.num_neurons, self.P))
      A = self.param('A', nn.initializers.normal(1/jnp.sqrt(self.num_neurons)), (self.num_neurons, self.P))
      y_pfr, s_tensor = LD_PFR_Module(num_neurons=self.num_neurons, P=self.P, latent_hidden_dim=self.latent_hidden_dim, name="PFR")(x, V, A) 

      global_E_context = jnp.mean(E, axis=0)
      
      s_flat = s_tensor.reshape(s_tensor.shape[0], -1) 

      y_carg, mod_factors, mod_raw_outputs, individual_corrector_outputs = CARG_Ablation_Module( # Get individual outputs
          num_neurons=self.num_neurons,
          rank=self.rank, 
          num_clusters=self.num_clusters,
          corrector_hidden_dim=self.corrector_hidden_dim,
          name="CARG_Ablation"
      )(c, s_flat)

      predicted_delta = y_nfm + y_pfr + y_carg
      predictions = x[:, -1, None, :] + predicted_delta
      return predictions

# Hyperparameters (from Logos II's code)
def _define_hyperparameters():
    return {
        'learning_rate': 1e-3,
        'rank': 32,
        'Qn': 12,
        'P': 32,
        'latent_hidden_dim': 64,
        'corrector_hidden_dim': 16,
        'num_clusters': 3
    }

# create_network function
def create_network(hparams):
    hp = hparams.copy(); hp.pop('learning_rate') 
    hp['num_neurons'] = 71721 
    return ModelInterface(LD_Net_CARG_Ablation, hparams=hp)

# compute_loss function (from Logos II's code, for consistency)
def compute_loss(predictions, targets, params, x):
    pred_delta = predictions - x[:, -1, None, :]; targ_delta = targets - x[:, -1, None, :]
    return jnp.mean(jnp.abs(pred_delta - targ_delta))

# create_optimizer function (from Logos II's code, for consistency)
def create_optimizer(learning_rate: float = 1e-3):
    return optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.adamw(learning_rate, weight_decay=WEIGHT_DECAY))
