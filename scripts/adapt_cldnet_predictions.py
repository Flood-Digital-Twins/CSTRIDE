#!/usr/bin/env python3
"""Stream public CLDNet --save-fields output into the historical dense scorer format."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from numpy.lib import format as npy_format
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from score_cldnet_baselines import open_prediction_member


def convert(source, grid_path, output, outside_mask=0.0):
    with np.load(grid_path, allow_pickle=False) as grid:
        height, width = grid['aggregate_mask'].shape
    with np.load(source, allow_pickle=False) as prediction:
        indices = prediction['cell_index'].astype(np.int64)
    archive, member, shape, dtype = open_prediction_member(source, 'fields.npy')
    try:
        if len(shape) != 3 or shape[1] != len(indices):
            raise ValueError('Expected fields (time, cells, channels) matching cell_index')
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= height * width):
            raise ValueError('Invalid or duplicate raster indices')
        output = Path(output)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as target:
            with target.open('pred_full.npy', 'w', force_zip64=True) as stream:
                npy_format.write_array_header_2_0(stream, {'descr': npy_format.dtype_to_descr(dtype), 'fortran_order': False, 'shape': (shape[0], height, width, shape[2])})
                count = shape[1] * shape[2] * dtype.itemsize
                for _ in range(shape[0]):
                    raw = member.read(count)
                    if len(raw) != count:
                        raise EOFError('Truncated CLDNet fields')
                    dense = np.full((height * width, shape[2]), outside_mask, dtype=dtype)
                    dense[indices] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
                    stream.write(dense.tobytes())
        output.with_suffix('.json').write_text(json.dumps({'input_filename': Path(source).name, 'shape': [shape[0], height, width, shape[2]], 'outside_aggregate_mask_value': outside_mask, 'qualification': 'CLDNet has no public prediction outside its aggregate mask. Filled cells are not model predictions; grids can differ from the C-STRIDE event-specific grid.'}, indent=2) + '\n')
    finally:
        member.close()
        archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--grid', type=Path, required=True, help='CLDNet data/illinois_grid/grid.npz')
    parser.add_argument('--output', type=Path, required=True, help='Use predictions_traj<ID>.npz for baseline scoring')
    parser.add_argument('--outside-mask', type=float, default=0.0)
    args = parser.parse_args()
    convert(args.input, args.grid, args.output, args.outside_mask)


if __name__ == '__main__':
    main()
