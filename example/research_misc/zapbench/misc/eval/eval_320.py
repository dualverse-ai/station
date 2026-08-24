import sys
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from flax.linen import initializers
import os, json # For saving analysis artifacts

# Copied from storage/synapse/dynamic_mixer_v1.py for self-contained submission
class GatingNetwork(nn.Module):
    num_experts: int
    @nn.compact
    def __call__(self, neuron_embeds): # neuron_embeds: (N, embed_dim)
        logits = nn.Dense(self.num_experts, name="gating_mlp")(neuron_embeds)
        return nn.softmax(logits, axis=-1) # (N, num_experts)

# Copied from storage/synapse/dynamic_mixer_v1.py for self-contained submission
class ExpertTemporalMLP(nn.Module):
    num_factors: int
    temporal_hidden: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, training: bool = False): # x: (B, 4, num_factors)
        B, T_in, F = x.shape
        x_flat = jnp.reshape(x, (B, T_in * F)) # (B, 4*num_factors)
        x_proj = nn.Dense(self.temporal_hidden)(x_flat)
        x_proj = nn.relu(x_proj)
        x_proj = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_proj)
        return nn.Dense(self.num_factors * 32)(x_proj) # (B, num_factors*32)

# Copied from storage/synapse/dynamic_mixer_v1.py for self-contained submission
class DynamicFunctionalMixer(nn.Module):
    num_experts: int
    neuron_embed_dim: int
    num_factors: int
    temporal_hidden: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, training: bool = False): # x: (B, 4, N)
        B, T_in, N = x.shape
        
        neuron_embeddings = self.param('neuron_embeddings', initializers.lecun_normal(), (N, self.neuron_embed_dim)) # (N, embed_dim)
        
        gates = GatingNetwork(num_experts=self.num_experts, name="gating_network")(neuron_embeddings) # (N, num_experts)
        
        expert_factors_list = []
        expert_temporal_mlps = []
        for i in range(self.num_experts):
            E_k = self.param(f'E_{i}', initializers.lecun_normal(), (N, self.neuron_embed_dim))
            A_k = self.param(f'A_{i}', initializers.lecun_normal(), (self.neuron_embed_dim, self.num_factors))
            W_k = E_k @ A_k # (N, num_factors)
            expert_factors_list.append(W_k)
            
            expert_temporal_mlps.append(ExpertTemporalMLP(num_factors=self.num_factors, temporal_hidden=self.temporal_hidden,
                                                         dropout_rate=self.dropout_rate, name=f"expert_temporal_mlp_{i}"))
        
        expert_z_hist = [jnp.einsum('b t n, n f -> b t f', x, W_k) for W_k in expert_factors_list] # List of (B,4,num_factors)
        
        expert_f_traj_flat = [expert_temporal_mlps[i](z_hist_k, training=training) for i, z_hist_k in enumerate(expert_z_hist)]
        
        expert_f_traj = [jnp.reshape(ft_flat, (B, self.num_factors, 32)) for ft_flat in expert_f_traj_flat] # List of (B, num_factors, 32)
        
        expert_y = [jnp.einsum('b f o, f n -> b o n', ft, W_k.T) for ft, W_k in zip(expert_f_traj, expert_factors_list)] # List of (B,32,N)

        gates_reshaped = gates[None, None, :, :] # (1, 1, N, num_experts)

        all_expert_y = jnp.stack(expert_y, axis=-1) # (B, 32, N, num_experts)
        
        y_global = jnp.sum(all_expert_y * gates_reshaped, axis=-1) # (B, 32, N)
        
        return y_global

# Copied from storage/synapse/hybrid_core_v1.py for self-contained submission
class LocalPath(nn.Module):
    local_hidden: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, training: bool = False): # x: (B, 4, N)
        B, T_in, N = x.shape
        x_bn4 = jnp.transpose(x, (0, 2, 1)) # (B, N, 4)
        x_flat = jnp.reshape(x_bn4, (B * N, 4)) # (B*N, 4)
        
        id_dim = 8 # hardcoded as per SOTA.
        id_embed = self.param('id_embed', initializers.lecun_normal(), (N, id_dim)) # (N, id_dim)
        feat_id = jnp.reshape(jnp.repeat(id_embed[None, :, :], B, axis=0), (B * N, id_dim)) # (B*N, id_dim)
        
        feats_local = jnp.concatenate([x_flat, feat_id], axis=-1) # (B*N, 4 + id_dim)

        z = nn.Dense(self.local_hidden)(feats_local)
        z = nn.relu(z)
        z = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z)
        y_local_flat = nn.Dense(32)(z) # Output 32 timesteps
        return jnp.reshape(y_local_flat, (B, N, 32)).transpose(0, 2, 1) # (B, 32, N)

class DFMHybridModel(nn.Module):
    num_experts: int = 8
    neuron_embed_dim: int = 16
    num_factors: int = 10
    temporal_hidden: int = 80
    local_hidden: int = 32
    dropout_rate: float = 0.02
    alpha: float = 0.5
    use_skip: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T_in, N = x.shape
        out_h = 32

        y_global = DynamicFunctionalMixer(num_experts=self.num_experts, 
                                          neuron_embed_dim=self.neuron_embed_dim, 
                                          num_factors=self.num_factors, 
                                          temporal_hidden=self.temporal_hidden, 
                                          dropout_rate=self.dropout_rate,
                                          name="DynamicFunctionalMixer_0")(x, training=training) # (B,32,N)
        
        y_local = LocalPath(local_hidden=self.local_hidden, 
                            dropout_rate=self.dropout_rate,
                            name="LocalPath_0")(x, training=training) # (B,32,N)

        if self.use_skip:
            skip_logits = self.param('skip_logits', initializers.zeros, (out_h,))
            w = nn.sigmoid(skip_logits)
            last_rep = jnp.repeat(x[:, -1:, :], out_h, axis=1) # (B,32,N)
            y_global = y_global + w[None, :, None] * last_rep

        y = self.alpha * y_global + (1.0 - self.alpha) * y_local
        
        gamma = self.param('gamma', initializers.ones, (N,))
        beta  = self.param('beta', initializers.zeros, (N,))
        y = y * gamma[None, None, :] + beta[None, None, :]
        return y

class Wrapper:
    def __init__(self, hparams):
        self.model = DFMHybridModel(
            num_experts=int(hparams.get('num_experts', 8)),
            neuron_embed_dim=int(hparams.get('neuron_embed_dim', 16)),
            num_factors=int(hparams.get('num_factors', 10)),
            temporal_hidden=int(hparams.get('temporal_hidden', 80)),
            local_hidden=int(hparams.get('local_hidden', 32)),
            dropout_rate=float(hparams.get('dropout_rate', 0.02)),
            alpha=float(hparams.get('alpha', 0.5)),
            use_skip=bool(hparams.get('use_skip', True)),
        )
        self.mutable = []
        self.needs_rng = True # Due to dropout

    def init(self, rng_key, dummy_input):
        keys = random.split(rng_key, 2)
        return self.model.init({'params': keys[0], 'dropout': keys[1]}, dummy_input, training=True)

    def apply(self, params, x, training=False, mutable=None, rngs=None):
        return self.model.apply(params, x, training=training, rngs={'dropout': rngs.get('dropout')} if rngs else None)

def _define_hyperparameters():
    return {
        'learning_rate': 1e-3,
        'num_experts': 8,
        'neuron_embed_dim': 16,
        'num_factors': 10,
        'temporal_hidden': 80,
        'local_hidden': 32,
        'dropout_rate': 0.02,
        'alpha': 0.5,
        'use_skip': True,
    }

def create_network(hparams):
    return Wrapper(hparams)

def compute_loss(predictions, targets, params, x):
    return jnp.mean(jnp.abs(predictions - targets))

def create_optimizer(learning_rate: float = 1e-3):
    return optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)

# Helper to find named leaves in param pytree
def _as_params_dict(params):
    if isinstance(params, dict) and 'params' in params:
        return params['params']
    return params

def _find_param(p, name: str):
    stack = [p]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == name and hasattr(v, 'shape'):
                    return v
                stack.append(v)
    return None

def calculate_gate_entropy(cluster_assignments, epsilon=1e-9):
    # cluster_assignments: (N, num_experts) - softmax outputs
    mean_dist = jnp.mean(cluster_assignments, axis=0) # (num_experts,)
    mean_dist_normalized = mean_dist / jnp.sum(mean_dist) # Ensure sums to 1
    positive_dist = mean_dist_normalized[mean_dist_normalized > epsilon]
    if positive_dist.size == 0:
        return 0.0
    return -jnp.sum(positive_dist * jnp.log2(positive_dist))

def calculate_effective_num_experts(gate_entropy):
    return jnp.power(2.0, gate_entropy)

def complete(params, opt_state, trial_data):
    vm = trial_data.get('val_mae', None); steps = trial_data.get('step_maes', None)
    if vm is not None: print(f"[Veritas_DFM_Analysis] Validation MAE (trial): {vm:.6f}")
    if steps is not None: print(f"[Veritas_DFM_Analysis] Step MAEs: {steps}")

    print("\n=== DFM CLUSTER ANALYSIS ===")
    p = _as_params_dict(params)

    try:
        # Extract necessary parameters for DFM reconstruction
        dfm_params = p['DynamicFunctionalMixer_0']
        gating_params = dfm_params['gating_network']
        neuron_embeddings = dfm_params['neuron_embeddings']
        
        num_experts = trial_data['hyperparameters']['num_experts']
        
        # Reconstruct GatingNetwork and apply to get cluster assignments
        gating_network_reconstructed = GatingNetwork(num_experts=num_experts, name="gating_network")
        cluster_assignments = gating_network_reconstructed.apply({'params': gating_params}, neuron_embeddings) # (N, num_experts)
        
        # Calculate cluster validity metrics
        hard_assign = jnp.argmax(cluster_assignments, axis=1)
        hard_cluster_counts = jnp.bincount(hard_assign, length=num_experts)

        gate_entropy = calculate_gate_entropy(cluster_assignments)
        effective_num_experts = calculate_effective_num_experts(gate_entropy)
        
        print(f"[DFM_Analysis] Effective Num Experts: {effective_num_experts:.4f}")
        print(f"[DFM_Analysis] Gate Entropy: {gate_entropy:.4f}")
        print(f"[DFM_Analysis] Hard Cluster Counts: {hard_cluster_counts}")

        # Prepare and save analysis artifact (simplified for now)
        analysis_results = {
            "eval_id": trial_data.get('eval_id', -1),
            "seed": trial_data.get('seed'),
            "hparams": trial_data.get('hyperparameters'),
            "val_mae": float(vm) if vm is not None else "N/A",
            "step_maes": {k: float(v) for k, v in steps.items()} if steps is not None else {},
            "custom_metrics": {
                "effective_num_experts": float(effective_num_experts),
                "gate_entropy": float(gate_entropy),
                "cluster_sizes": [int(c) for c in hard_cluster_counts]
            },
            "notes": "Initial cluster validity analysis for DFM hybrid replication."
        }

        analysis_dir = os.path.abspath(f'storage/{trial_data["agent_name"]}/analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        filename = f"analysis_dfm_eval_{trial_data.get('eval_id', 'unknown')}_{trial_data.get('seed')}.json"
        filepath = os.path.join(analysis_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"[DFM_Analysis] Successfully saved contract-compliant analysis artifact to {filepath}")

    except Exception as e:
        print(f"[DFM_Analysis] ERROR in complete() function: {e}")

BATCH_SIZE = 16
