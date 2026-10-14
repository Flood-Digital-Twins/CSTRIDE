import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cstride.chicago import MinMaxNormalizer, chicago_split_indices, make_windows
from cstride.forecast_observations import SensorObservationLSTM, make_observation_prediction_tensors
from cstride.forecast_latents import LatentPredictionLSTM
from cstride.models import CSTRIDE
from cstride.runtime import load_state_dict
from cstride.soap import SOAP

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_checkpoint_integrity():
    for name, entry in json.loads((ROOT / 'checkpoints/manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry['sha256']


@pytest.mark.parametrize('run', sorted((p for p in (ROOT / 'checkpoints').iterdir() if p.is_dir())), ids=lambda p: p.name)
def test_reference_checkpoints(run):
    config = yaml.safe_load((run / 'config.yaml').read_text())
    basics = config['basics']
    rain = basics.get('include_precipitation_input', True)
    sensors = basics['num_sensors']
    lag = basics['lag']
    terrain = basics.get('terrain_conditioning', True)
    model = CSTRIDE(input_size=sensors + 507 * rain, output_size=1,
        coord_size=5 if terrain else 2, device=torch.device('cpu'), **config['model'])
    for checkpoint in ('best_model.ckpt', 'final_model.ckpt'):
        model.load_state_dict(load_state_dict(run / checkpoint))
        model.eval()
        with torch.no_grad():
            output, latent = model(torch.zeros(1, lag, sensors + 507 * rain), torch.zeros(1, 7, 5 if terrain else 2))
        assert output.shape == (1, 7, 1) and torch.isfinite(output).all()
        assert latent.shape == (1, 256)
    for path in run.glob('sensor_observation*.ckpt'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        obs = SensorObservationLSTM(**{k: state[k] for k in ['input_size', 'sensor_size', 'hidden_size', 'num_layers', 'dropout']})
        obs.load_state_dict(state['model_state_dict'])
        assert torch.isfinite(obs(torch.zeros(1, lag, state['input_size']))).all()
    for path in run.glob('latent_prediction*.ckpt'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        latent_model = LatentPredictionLSTM(**{k: state[k] for k in ['latent_size', 'hidden_size', 'num_layers', 'dropout']})
        latent_model.load_state_dict(state['model_state_dict'])
        assert torch.isfinite(latent_model(torch.zeros(1, lag, state['latent_size']))).all()


def test_time_alignment_and_normalization():
    sequence = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    windows = make_windows(sequence, 3)
    assert windows[:, -1, 0].tolist() == [2, 3, 4, 5, 6]
    x, y, forcing = make_observation_prediction_tensors(sequence, [0], 8, 3, 1, 1, 2)
    assert x[0, :, 0].tolist() == [0, 0, 0]
    assert y[0, :, 0].tolist() == [1, 2]
    assert y[-1, :, 0].tolist() == [6, 7]
    assert forcing.shape == (6, 2, 0)
    normalizer = MinMaxNormalizer([0, 5], [10, 5])
    data = np.array([[3, 5], [8, 5]], dtype=np.float32)
    np.testing.assert_allclose(normalizer.inverse_transform(normalizer.transform(data)), data)
    train, valid, test = chicago_split_indices(114)
    assert len(train) == 100 and valid.tolist() == list(range(100, 110))
    assert test.tolist() == [100, 101, 102, 113]


def test_optimizer_updates_trainable_weights_only():
    torch.manual_seed(7)
    model = CSTRIDE(2, 1, hidden_size=4, hidden_layers=1, coord_size=2,
                               fmmnn_depth=2, fmmnn_width=8, fmmnn_rank=3)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = SOAP(model.parameters(), lr=0.002)
    for _ in range(2):
        optimizer.zero_grad()
        pred, _ = model(torch.ones(2, 3, 2), torch.ones(2, 4, 2))
        pred.square().mean().backward()
        optimizer.step()
    assert any(not torch.equal(before[k], p) for k, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[k], p) for k, p in model.named_parameters() if not p.requires_grad)


def test_storm_split_is_independent_of_manifest_order():
    split = json.loads((ROOT / 'splits/chicago_split.json').read_text())
    ids = sorted(set(split['train_files'] + split['checkpoint_monitor_files'] + split['test'] + split['not_used']), reverse=True)
    entries = [dict(traj_id=i) for i in ids]
    train, valid, test = chicago_split_indices(entries)
    assert [entries[i]['traj_id'] for i in train] == split['train_files']
    assert [entries[i]['traj_id'] for i in valid] == split['checkpoint_monitor_files']
    assert [entries[i]['traj_id'] for i in test] == split['test']


def test_chunked_decoder_matches_full_forward():
    from cstride.released import encode, decode_chunks
    torch.manual_seed(11)
    model = CSTRIDE(2, 1, hidden_size=4, hidden_layers=1, coord_size=2,
                              fmmnn_depth=2, fmmnn_width=8, fmmnn_rank=3).eval()
    history = torch.randn(1, 3, 2)
    coords = np.random.default_rng(11).random((11, 2), dtype=np.float32)
    with torch.inference_mode():
        expected, _ = model(history, torch.tensor(coords)[None])
        actual = np.concatenate([v for _, v in decode_chunks(model, encode(model, history), coords, 4)])
    np.testing.assert_allclose(actual, expected[0, :, 0].numpy(), atol=1e-6, rtol=1e-5)


def test_public_baseline_adapter_preserves_raster_order(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('cldnet_adapter', ROOT / 'scripts/adapt_cldnet_predictions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = np.arange(12, dtype=np.float16).reshape(2, 3, 2)
    source, grid, output = [tmp_path / name for name in ('public.npz', 'grid.npz', 'predictions_traj107.npz')]
    np.savez(source, fields=fields, cell_index=[5, 0, 3])
    np.savez(grid, aggregate_mask=np.zeros((2, 3), dtype=bool))
    module.convert(source, grid, output)
    with np.load(output, allow_pickle=False) as result:
        dense = result['pred_full'].reshape(2, 6, 2)
        np.testing.assert_array_equal(dense[:, [5, 0, 3]], fields)
        np.testing.assert_array_equal(dense[:, [1, 2, 4]], 0)


def test_event_id_only_inputs_preserve_forecast_origins_and_seeds():
    import sys
    from cstride.chicago import chicago_event_seed_indices
    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_chicago_future_precipitation import select_section
    from plot_chicago_precipitation_mismatch import parse_evaluation_starts
    split = json.loads((ROOT / "splits/chicago_split.json").read_text())
    entries = [{"traj_id":i} for i in reversed(sorted(split["train"] + split["test"]))]
    train, monitor, test = chicago_split_indices(entries)
    assert {entries[i]["traj_id"] for i in train} == set(split["train"])
    assert [entries[i]["traj_id"] for i in test] == split["test"]
    seeds = chicago_event_seed_indices(entries)
    assert seeds[test].tolist() == [100,101,102,113]
    selected = select_section(ROOT / "checkpoints/full", True, 0.0, True)
    origins = parse_evaluation_starts(selected, entries)
    legacy_origins = parse_evaluation_starts(selected)
    assert len(origins) == 4 and sum(map(len, origins.values())) == 40
    assert {int(seeds[i]):v for i,v in origins.items()} == legacy_origins
