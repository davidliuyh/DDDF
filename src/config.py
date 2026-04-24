"""Configuration for the WN IC→Residual pipeline."""

import os

# ── Storage paths ────────────────────────────────────────────────────────────
pscratch   = '/pscratch/sd/l/liuyh15/DDDF/ZA'
data_path  = f'{pscratch}/data'
model_path = f'{pscratch}/models'

# ── Simulation defaults ───────────────────────────────────────────────────────
L          = 1
N_p        = 256
boxsize    = 1000.0 * L   # Mpc/h
Omega_m    = 0.3175
MAS        = 'CIC'
threads    = 32

final_snapshot_z = 0.0


def ensure_filter_dirs(data_dir, model_dir=None):
    """Ensure data/model subdirectories exist."""
    if model_dir is None:
        model_dir = data_dir
    os.makedirs(f'{data_path}/{data_dir}', exist_ok=True)
    os.makedirs(f'{model_path}/{model_dir}', exist_ok=True)


# ── Override directories and epochs ──────────────────────────────────────────
data_dir   = 'baseline'
model_dir  = 'psi_vec_v3'
filter_dir = data_dir                           # backward-compat alias

epochs     = 736   # smoke-test; bump later for real training

ensure_filter_dirs(data_dir, model_dir)

# ── Snapshot helpers ──────────────────────────────────────────────────────────
def snapshot_paths(realization, N_p=N_p, L=L):
    if N_p == 128:
        return [
            f'../FastPM/L{L}N{N_p}fnl0r1000{realization + 1}/dm/dm_0.0100/1',
            f'../FastPM/L{L}N{N_p}fnl0r1000{realization + 1}/dm/dm_1.0000/1',
        ]
    elif N_p == 256:
        return [
            f'/pscratch/sd/l/liuyh15/Quijote/fiducial_LR/{realization}/ICs/ics',
            f'/pscratch/sd/l/liuyh15/Quijote/fiducial_LR/{realization}/snapdir_004/snap_004',
        ]
    else:
        return [
            f'/pscratch/sd/l/liuyh15/Quijote/fiducial/{realization}/ICs/ics',
            f'/pscratch/sd/l/liuyh15/Quijote/fiducial/{realization}/snapdir_004/snap_004',
        ]


def snapshot_format(N_p=N_p):
    return 'bigfile' if N_p == 128 else 'gadget'


# ── Training hyperparameters ──────────────────────────────────────────────────
train_realizations  = list(range(16))
patch_size          = 20
padding             = 2
overlap             = 0.0
rotate              = True
checkpoint_interval = 1

# ── GAN hyperparameters ───────────────────────────────────────────────────────
batch_size            = 1024
gan_lambda_pixel      = 5.0
gan_lr_g              = 1e-4
gan_lr_d              = 5e-5
gan_n_disc_layers     = 3
gan_lambda_fm         = 20.0
gan_lambda_gp         = 10.0
gan_d_update_interval = 3

# ── Inference hyperparameters ─────────────────────────────────────────────────
infer_patch_size  = 20
infer_padding     = 2
infer_overlap     = 0.0
infer_batch_size  = 32
infer_epochs      = epochs   # default: use the training epochs
infer_checkpoint  = None     # None → auto-derive from training model_name + infer_epochs
k_cut              = 0.005
k_width            = 0.0025

# ── Output path helpers ───────────────────────────────────────────────────────
def _resolve_data_dir(data_dir=None, filter_dir=None):
    if data_dir is not None:
        return data_dir
    if filter_dir is not None:
        return filter_dir
    return globals()['data_dir']


def _resolve_model_dir(model_dir=None, filter_dir=None):
    if model_dir is not None:
        return model_dir
    if filter_dir is not None:
        return filter_dir
    return globals()['model_dir']


def best_fit_coef_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/baseline-coefL{L}N{N_p}-{realization}.npz'


def best_fit_avg_coef_path(data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/baseline-coef-avgL{L}N{N_p}.npz'


def best_fit_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/baselineL{L}N{N_p}-{realization}.npz'


def best_fit_delta_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/baseline-deltaL{L}N{N_p}-{realization}.npz'


def target_delta_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/target-deltaL{L}N{N_p}-{realization}.npz'


def realization_tag(realizations):
    if isinstance(realizations, (list, tuple)):
        return '+'.join(str(r) for r in realizations)
    return str(realizations)


def training_data_path(realizations, ps=patch_size, p=padding, ov=overlap,
                       rot=rotate, N_p=N_p, data_dir=None, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    tag = f'IC2RES-N{N_p}PS{ps}P{p}O{int(100 * ov):02d}Rotate{rot}-{realization_tag(realizations)}'
    return f'{data_path}/{ddir}/training-data-{tag}.npz'


def gan_model_name(realizations, ps=patch_size, p=padding,
                   rot=rotate, N_p=N_p, model_dir=None, filter_dir=None):
    mdir = _resolve_model_dir(model_dir=model_dir, filter_dir=filter_dir)
    rtag = realization_tag(realizations)
    return f'{model_path}/{mdir}/gan-IC2RES-N{N_p}PS{ps}P{p}Rotate{rot}-{rtag}'


# ── WN-specific path helpers ─────────────────────────────────────────────────

def wn_dir_path(realization, N_p=N_p):
    """Return the white-noise directory for a given realization."""
    catalog = 'fiducial_LR' if N_p <= 256 else 'fiducial'
    return f'/pscratch/sd/l/liuyh15/Quijote/{catalog}/{realization}/wn'


def wn_psi1_path(realization, N_p=N_p):
    """Return explicit path to the psi1 file for a given realization."""
    return f'{wn_dir_path(realization, N_p)}/psi1_grid_z127_seed{realization:03d}_n{N_p}.npz'


def wn_qinit_path(realization, N_p=N_p):
    """Return explicit path to the q_init file for a given realization."""
    return f'{wn_dir_path(realization, N_p)}/q_init_seed{realization:03d}_n{N_p}.npz'


def final_snapshot_path(realization, N_p=N_p):
    """Return only the z=0 snapshot path."""
    return snapshot_paths(realization, N_p)[1]


# ── Vector-Ψ pipeline config ─────────────────────────────────────────────────

# Versioned vector-GAN presets, mirroring config.py style (gan_v1/gan_v2/...).
vec_gan_v1 = {
    'vec_data_dir': 'psi_vec',
    'vec_model_dir': 'psi_vec_v1',
    'vec_batch_size': 512,
    'vec_rotate': False,
    'vec_unet_base_channels': 16,
    'epochs': 736,
    'gan_lambda_pixel': 5.0,
    'gan_lr_g': 1e-4,
    'gan_lr_d': 5e-5,
    'gan_n_disc_layers': 3,
    'gan_lambda_fm': 20.0,
    'gan_lambda_gp': 10.0,
    'gan_d_update_interval': 3,
}

vec_gan_v2 = {
    'vec_data_dir': 'psi_vec',
    'vec_model_dir': 'psi_vec_v2',
    'vec_batch_size': 512,
    'vec_rotate': False,
    'vec_unet_base_channels': 32,
    'epochs': 100,
    'gan_lambda_pixel': 5.0,
    'gan_lr_g': 1e-4,
    'gan_lr_d': 5e-5,
    'gan_n_disc_layers': 3,
    'gan_lambda_fm': 20.0,
    'gan_lambda_gp': 10.0,
    'gan_d_update_interval': 3,
    'gan_use_multiscale_disc': False,
    'gan_disc_base_channels': 32,
}

vec_gan_v3 = {
    'vec_data_dir': 'psi_vec',
    'vec_model_dir': 'psi_vec_v3',
    'vec_batch_size': 512,
    'vec_rotate': False,
    'vec_unet_base_channels': 32,
    'epochs': 300,
    'gan_lambda_pixel': 5.0,
    'gan_lr_g': 1e-4,
    'gan_lr_d': 1e-4,
    'gan_n_disc_layers': 3,
    'gan_lambda_fm': 20.0,
    'gan_lambda_gp': 10.0,
    'gan_d_update_interval': 1,
    'gan_use_multiscale_disc': True,
    'gan_disc_base_channels': 64,
}

# Activate one vector preset here.
active_vec_gan_defaults = vec_gan_v3

vec_data_dir  = active_vec_gan_defaults['vec_data_dir']
vec_model_dir = active_vec_gan_defaults['vec_model_dir']
vec_batch_size = active_vec_gan_defaults['vec_batch_size']
vec_rotate     = active_vec_gan_defaults['vec_rotate']
vec_unet_base_channels = active_vec_gan_defaults['vec_unet_base_channels']

# Keep training GAN hyperparameters synchronized with the active vector preset.
epochs                = active_vec_gan_defaults['epochs']
gan_lambda_pixel      = active_vec_gan_defaults['gan_lambda_pixel']
gan_lr_g              = active_vec_gan_defaults['gan_lr_g']
gan_lr_d              = active_vec_gan_defaults['gan_lr_d']
gan_n_disc_layers     = active_vec_gan_defaults['gan_n_disc_layers']
gan_lambda_fm         = active_vec_gan_defaults['gan_lambda_fm']
gan_lambda_gp         = active_vec_gan_defaults['gan_lambda_gp']
gan_d_update_interval = active_vec_gan_defaults['gan_d_update_interval']
gan_use_multiscale_disc = active_vec_gan_defaults['gan_use_multiscale_disc']
gan_disc_base_channels  = active_vec_gan_defaults['gan_disc_base_channels']
infer_epochs          = epochs

ensure_filter_dirs(vec_data_dir, vec_model_dir)


def vec_training_data_path(realizations, ps=patch_size, p=padding, ov=overlap,
                           rot=vec_rotate, N_p=N_p):
    tag = f'VEC-N{N_p}PS{ps}P{p}O{int(100 * ov):02d}Rotate{rot}-{realization_tag(realizations)}'
    return f'{data_path}/{vec_data_dir}/training-data-{tag}.npz'


def vec_gan_model_name(realizations, ps=patch_size, p=padding,
                       rot=vec_rotate, N_p=N_p):
    rtag = realization_tag(realizations)
    return f'{model_path}/{vec_model_dir}/gan-VEC-N{N_p}PS{ps}P{p}Rotate{rot}-{rtag}'
