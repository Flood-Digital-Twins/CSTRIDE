"""Exercise real CLI workflows against a tiny synthetic Chicago-format dataset."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
import torch
import yaml


@pytest.mark.parametrize("event_ids_only", [False, True])
def test_training_and_evaluation_workflows(tmp_path, event_ids_only):
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(10)
    arrays = {'state': rng.random((16, 8, 3), dtype=np.float32),
              'coords': rng.random((8, 2), dtype=np.float32),
              'precip': rng.random((16, 2), dtype=np.float32),
              'elevation': rng.random(8, dtype=np.float32),
              'manning': rng.random(8, dtype=np.float32),
              'slope': rng.random(8, dtype=np.float32)}
    for name, array in arrays.items():
        np.save(data / f'{name}.npy', array)
    split = json.loads(Path('splits/chicago_split.json').read_text())
    ids = sorted(set(split['train_files'] + split['checkpoint_monitor_files'] + split['test'] + split['not_used']))
    if event_ids_only:
        ids = sorted(split['train'] + split['test'])
    entries = [dict(traj_id=i, **{f'{name}_file': f'{name}.npy' for name in arrays}) for i in ids]
    (data / 'manifest.json').write_text(json.dumps({'time_steps': 16, 'files': entries}))
    sensor_path = tmp_path / 'sensors.pt'
    torch.save({i: np.array([0, 2]) for i in ids}, sensor_path)
    config = yaml.safe_load(Path('configs/next_step/full.yaml').read_text())
    config['basics'].update(data_root=str(data), device='cpu', num_sensors=2, lag=3, sensor_idx_path=str(sensor_path))
    config['model'].update(hidden_size=4, hidden_layers=1, fmmnn_depth=2, fmmnn_width=8, fmmnn_rank=3)
    config['training'].update(epochs=10, num_points=4, batch_size=2048, num_workers=0,
                              save_path=str(tmp_path / 'next'), log_interval=100)
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(tmp_path / 'mpl'))
    def run(module, *args):
        result = subprocess.run([sys.executable, f'scripts/{module}.py', *map(str, args)],
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    run('train_next_step', '--config', cfg)
    output = next((tmp_path / 'next').glob('training_*'))
    assert (output / 'best_model.ckpt').exists()
    run('evaluate_next_step', '--path', output, '--data-root', data, '--device', 'cpu', '--num-workers', 0)
    run('forecast_observations', '--path', output, '--horizon', 2, '--num_starts', 1,
        '--obs_epochs', 1, '--obs_batch_size', 2048, '--obs_hidden_size', 4, '--obs_layers', 1)
    # Run a second time to cover reuse of the trained auxiliary checkpoint.
    run('forecast_observations', '--path', output, '--horizon', 2, '--num_starts', 1,
        '--obs_hidden_size', 4, '--obs_layers', 1)
    run('forecast_latents', '--path', output, '--horizon', 2, '--num_starts', 1,
        '--latent_epochs', 1, '--latent_batch_size', 2048, '--latent_hidden_size', 4, '--latent_layers', 1)
    config['forecast'] = {'horizon': 2, 'lookback_steps': 0, 'discount_factor': 0.8, 'observed_time_fraction': 1.0}
    config['training'].update(epochs=1, eval_interval=1, save_path=str(tmp_path / 'ar'))
    cfg.write_text(yaml.safe_dump(config))
    run('train_autoregressive', '--config', cfg)
    ar_output = next((tmp_path / 'ar').glob('training_*'))
    run('evaluate_autoregressive', '--path', ar_output, '--ar_horizon', 2, '--ar_num_starts', 1)
    assert (ar_output / 'results.txt').exists()
