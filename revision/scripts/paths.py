"""Where the revision scripts read their inputs and write their outputs.

Every location can be set with an environment variable; the defaults follow this repository's layout.

  CSTRIDE_ROOT           this repository (code/, checkpoints/, splits/, data/). Default: two levels above this file.
  REVISION_OUTPUTS       output root. Default: revision/results, so a rerun overwrites the published tables
                         and `git diff` shows any change.
  CSTRIDE_ARRAYS         prepared Chicago arrays written by preprocessing/ (114 trajectories). Default: data/Chicago.
  CSTRIDE_PAPER_RESULTS  paper products written by scripts/generate_paper_results.py. Default: outputs/paper.
  SYNXFLOW_SIMS          dense SynxFlow simulations, one folder per event with flow_variables.npy (not distributed).
  CLDNET_ROOT            a clone of github.com/Flood-Digital-Twins/CLDNet (needs data/illinois_grid/grid.npz).
  USGS_2013_GAUGES       the April 2013 USGS validation folder of the CLDNet study: gages.geojson,
                         USGS_2013-04-15_2013-04-23/, dem_5070.tif, truth_traj120.npy and output/ (the last three
                         are not in the public CLDNet repository). Default: <CLDNET_ROOT>/data/usgs_2013_validation.
  LDENSF_PREDICTIONS     folder with CLDNet + LD-EnSF predictions_traj{107,108,109,120}.npz (used by d1 only).
"""
import os
from pathlib import Path


def _path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


CSTRIDE_ROOT = _path('CSTRIDE_ROOT', Path(__file__).resolve().parents[2])
OUTPUTS = _path('REVISION_OUTPUTS', CSTRIDE_ROOT / 'revision' / 'results')
ARRAYS = _path('CSTRIDE_ARRAYS', CSTRIDE_ROOT / 'data' / 'Chicago')
PAPER_RESULTS = _path('CSTRIDE_PAPER_RESULTS', CSTRIDE_ROOT / 'outputs' / 'paper')
SIMS = _path('SYNXFLOW_SIMS', CSTRIDE_ROOT / 'data' / 'synxflow_sims')
CLDNET = _path('CLDNET_ROOT', CSTRIDE_ROOT.parent / 'CLDNet')
USGS_2013 = _path('USGS_2013_GAUGES', CLDNET / 'data' / 'usgs_2013_validation')
LDENSF = _path('LDENSF_PREDICTIONS', CSTRIDE_ROOT / 'data' / 'ldensf_predictions')
