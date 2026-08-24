## SOTA Tuning: Feedback Timestep Sweep (t=31, Bugfix v2)
## 3dbc6c7c-0f0c-40ce-b7d6-a549f8b765fe

import flax.linen as nn
import jax.numpy as jnp

class FeedbackTuningSOTA(nn.Module):
    # BUGFIX: Moved non-default argument to the beginning of the definition.
    feedback_timestep: int 
    rank_k: int = 72
    time_basis_M: int = 16
    time_hidden: int = 64
    cnn_kernel_size: int = 5
    output_horizon: int = 32
    input_horizon: int = 4
    ltcnn_features: int = 4
    ltcnn_kernel_size: int = 2

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T_in, N = x.shape
        W_in = self.param('W_in', nn.initializers.lecun_normal(), (N, self.rank_k))
        W_out = jnp.transpose(W_in)
        
        H = jnp.einsum('btn,nk->btk', x, W_in)
        C_0 = self.run_time_head(H, is_feedback=False)
        Bmat = self.param('B', nn.initializers.lecun_normal(), (self.time_basis_M, self.output_horizon))
        Y_lat_0 = jnp.einsum('bkm,mt->bkt', C_0, Bmat); Y_0 = jnp.einsum('bkt,kn->btn', Y_lat_0, W_out)
        spatial_cnn = nn.Conv(self.output_horizon, (self.cnn_kernel_size,), padding='SAME', name="SpatialCNN")
        Yref_0 = jnp.transpose(nn.relu(spatial_cnn(jnp.transpose(Y_0, (0, 2, 1)))), (0, 2, 1))
        
        feedback_signal = Yref_0[:, self.feedback_timestep, :]; H_feedback = jnp.einsum('bn,nk->bk', feedback_signal, W_in)
        H_combined = H + H_feedback[:, None, :]
        C_1 = self.run_time_head(H_combined, is_feedback=True)
        Y_lat_1 = jnp.einsum('bkm,mt->bkt', C_1, Bmat); Y_1 = jnp.einsum('bkt,kn->btn', Y_lat_1, W_out)
        Yref_1 = jnp.transpose(nn.relu(spatial_cnn(jnp.transpose(Y_1, (0, 2, 1)))), (0, 2, 1))
        
        z_query = jnp.mean(C_1, axis=2)
        attention_head = nn.Dense(T_in, name='attention_head')
        attention_logits = attention_head(z_query)
        attention_weights = nn.softmax(attention_logits, axis=-1)
        attentive_residual_base = jnp.einsum('bt,btn->bn', attention_weights, x)
        gamma_logits = self.param('gamma_logits', nn.initializers.zeros, (self.output_horizon,))
        gamma = nn.sigmoid(gamma_logits)
        residual = jnp.einsum('t,bn->btn', gamma, attentive_residual_base)
        
        s = self.param('calib_scale', nn.initializers.ones, (1, 1, N)); b = self.param('calib_bias', nn.initializers.zeros, (1, 1, N))
        return (Yref_1 + residual) * s + b

    def run_time_head(self, H, is_feedback):
        B, T_in, k = H.shape; name_suffix = "_Feedback" if is_feedback else "_Initial"
        H_reshaped = H.reshape((B * k, T_in, 1))
        C_processed = nn.Conv(self.ltcnn_features, (self.ltcnn_kernel_size,), padding='VALID', name=f"LTCNN{name_suffix}")(H_reshaped)
        C_flat = C_processed.reshape((B * k, -1))
        C = nn.Dense(self.time_hidden, name=f"Dense1{name_suffix}")(C_flat); C = nn.relu(C)
        C = nn.Dense(self.time_basis_M, name=f"Dense2{name_suffix}")(C)
        return C.reshape((B, k, self.time_basis_M))

def _find_param(params, name):
    if isinstance(params, dict):
        if name in params: return params[name]
        for v in params.values():
            p = _find_param(v, name)
            if p is not None: return p
    return None
def compute_loss(predictions, targets, params, x):
    early_strength=0.35; tau=8.0; err = jnp.abs(predictions - targets); T = predictions.shape[1]; t = jnp.arange(T, dtype=err.dtype); w = 1.0 + early_strength * jnp.exp(-t / tau); w = w / jnp.mean(w); mae = jnp.mean(err * w[None, :, None]); lam_calib = 5e-7; reg_calib = 0.0; p_inner = params.get('params', params); s = _find_param(p_inner, 'calib_scale'); b = _find_param(p_inner, 'calib_bias');
    if s is not None and b is not None: reg_calib = lam_calib * (jnp.mean((s - 1.0) ** 2) + jnp.mean(b ** 2))
    lam_tv = 3e-7; reg_tv = 0.0; W_in = _find_param(p_inner, 'W_in');
    if W_in is not None: tv = jnp.mean((W_in[1:, :] - W_in[:-1, :])**2); reg_tv = lam_tv * tv
    return mae + reg_calib + reg_tv

BASE_SEED = 42
BATCH_SIZE = 32
def _define_hyperparameters():
    return { 'learning_rate': 0.001, 'feedback_timestep': 31 }
class ModelWrapper:
    def __init__(self, hparams):
        model_params = {k:v for k,v in hparams.items() if k != 'learning_rate'}
        self.model = FeedbackTuningSOTA(**model_params)
        self.mutable = []; self.needs_rng = False
    def init(self, rng, dummy): return self.model.init({'params': rng}, dummy)
    def apply(self, params, x, **kwargs): return self.model.apply(params, x, **kwargs)
def create_network(hparams): return ModelWrapper(hparams)
