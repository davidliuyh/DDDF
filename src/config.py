"""
Central configuration for the IC->Residual pipeline.

Runtime settings (realization, N_p, data_dir, model_dir) are set in the notebook;
everything here provides defaults and path-building helpers.
"""

import os

# ── Storage paths ────────────────────────────────────────────────────────────
pscratch  = '/pscratch/sd/l/liuyh15/DDDF'
data_path = f'{pscratch}/data'
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
    """Ensure data/model subdirectories exist (supports split directories)."""
    if model_dir is None:
        model_dir = data_dir
    os.makedirs(f'{data_path}/{data_dir}', exist_ok=True)
    os.makedirs(f'{model_path}/{model_dir}', exist_ok=True)


def _resolve_data_dir(data_dir=None, filter_dir=None):
    """Resolve data subdirectory while preserving legacy filter_dir support."""
    if data_dir is not None:
        return data_dir
    if filter_dir is not None:
        return filter_dir
    return globals()['data_dir']


def _resolve_model_dir(model_dir=None, filter_dir=None):
    """Resolve model subdirectory while preserving legacy filter_dir support."""
    if model_dir is not None:
        return model_dir
    if filter_dir is not None:
        return filter_dir
    return globals()['model_dir']

def init_redshift(N_p):
    """Initial snapshot redshift: 99 for FastPM, 127 for Quijote."""
    return 99.0 if N_p == 128 else 127.0

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
train_realizations  = list(range(16))   # realizations used for training
patch_size          = 20
padding             = 2
overlap             = 0.0
rotate              = True
batch_size          = 512
epochs              = 25
learning_rate       = 1e-3
weight_decay        = 1e-5
checkpoint_interval = 1

# ── Training mode ─────────────────────────────────────────────────────────────
train_mode          = 'gan'  # 'unet' | 'gan'

# ── GAN hyperparameters (used when train_mode = 'gan') ────────────────────────
# Preserve original GAN defaults for easy comparison / rollback.
gan_v1 = {
    'data_dir': 'tophat0.4',
    'model_dir': 'tophat0.4v1',
    'overlap': 0.0,
    'batch_size': 512,
    'epochs': 25,
    'gan_lambda_pixel': 5.0,
    'gan_lr_g': 1e-4,
    'gan_lr_d': 5e-5,
    'gan_n_disc_layers': 3,
    'gan_lambda_fm': 20.0,
    'gan_d_update_interval': 3,
    'gan_use_spectral_norm': False,
    'infer_overlap': 0.0,
}

gan_v2 = {
    'data_dir': 'tophat0.4v2',
    'model_dir': 'tophat0.4v2',
    'overlap': 0.0,
    'batch_size': 512,
    'epochs': 25,
    'gan_lambda_pixel': 5.0,
    'gan_lr_g': 1e-4,
    'gan_lr_d': 5e-5,
    'gan_n_disc_layers': 3,
    'gan_lambda_fm': 20.0,
    'gan_d_update_interval': 3,
    'gan_use_spectral_norm': False,
    'infer_overlap': 0.0,
}

active_gan_defaults = gan_v1

# Active subdirs and hyperparameters are always sourced from the selected preset.
data_dir              = active_gan_defaults.get('data_dir', active_gan_defaults.get('filter_dir'))
model_dir             = active_gan_defaults.get('model_dir', data_dir)
# Backward-compat alias. New code should use data_dir/model_dir.
filter_dir            = data_dir
overlap               = active_gan_defaults['overlap']
batch_size            = active_gan_defaults['batch_size']
epochs                = active_gan_defaults['epochs']
gan_lambda_pixel      = active_gan_defaults['gan_lambda_pixel']
gan_lr_g              = active_gan_defaults['gan_lr_g']
gan_lr_d              = active_gan_defaults['gan_lr_d']
gan_n_disc_layers     = active_gan_defaults['gan_n_disc_layers']
gan_lambda_fm         = active_gan_defaults['gan_lambda_fm']
gan_d_update_interval = active_gan_defaults['gan_d_update_interval']
gan_use_spectral_norm = active_gan_defaults['gan_use_spectral_norm']

# ── Inference hyperparameters ─────────────────────────────────────────────────
infer_patch_size  = 20
infer_padding     = 2
infer_overlap     = 0.0
infer_epochs      = 25    # which epoch checkpoint to load (when infer_checkpoint is None)
infer_checkpoint  = None  # None → auto-derive from training model_name + infer_epochs
                          # str  → explicit path to a .pth checkpoint

# Create required directories as soon as config is imported.
ensure_filter_dirs(data_dir, model_dir)

# ── Output path helpers (match existing filenames on disk) ────────────────────
def best_fit_coef_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/best-fit-coefL{L}N{N_p}-{realization}.npz'

def best_fit_avg_coef_path(data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/best-fit-coef-avgL{L}N{N_p}.npz'

def best_fit_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/best-fitL{L}N{N_p}-{realization}.npz'

def best_fit_delta_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/best-fit-deltaL{L}N{N_p}-{realization}.npz'

def target_delta_path(realization, data_dir=None, L=L, N_p=N_p, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    return f'{data_path}/{ddir}/target-deltaL{L}N{N_p}-{realization}.npz'

def realization_tag(realizations):
    """Convert int or list of ints to a compact string tag, e.g. [0,1,2] → '0+1+2'."""
    if isinstance(realizations, (list, tuple)):
        return '+'.join(str(r) for r in realizations)
    return str(realizations)

def training_data_path(realizations, ps=patch_size, p=padding, ov=overlap,
                       rot=rotate, N_p=N_p, data_dir=None, filter_dir=None):
    ddir = _resolve_data_dir(data_dir=data_dir, filter_dir=filter_dir)
    tag = f'IC2RES-N{N_p}PS{ps}P{p}O{int(100 * ov):02d}Rotate{rot}-{realization_tag(realizations)}'
    return f'{data_path}/{ddir}/training-data-{tag}.npz'

def unet_model_name(realizations, ps=patch_size, p=padding,
                    rot=rotate, N_p=N_p, model_dir=None, filter_dir=None):
    mdir = _resolve_model_dir(model_dir=model_dir, filter_dir=filter_dir)
    rtag = realization_tag(realizations)
    return f'{model_path}/{mdir}/unet-IC2RES-N{N_p}PS{ps}P{p}Rotate{rot}-{rtag}'


def gan_model_name(realizations, ps=patch_size, p=padding,
                   rot=rotate, N_p=N_p, model_dir=None, filter_dir=None):
    mdir = _resolve_model_dir(model_dir=model_dir, filter_dir=filter_dir)
    rtag = realization_tag(realizations)
    return f'{model_path}/{mdir}/gan-IC2RES-N{N_p}PS{ps}P{p}Rotate{rot}-{rtag}'
