# Third-party notices

- `code/cstride/modulation.py` retains `LatentToModulation` and `Bias` from the project's vendored CORAL `coral/siren.py`, whose header also credits [EmilienDupont/coin](https://github.com/EmilienDupont/coin). The local [CORAL](https://github.com/LouisSerrano/coral) MIT license is retained at [licenses/CORAL.txt](licenses/CORAL.txt).
- `code/cstride/soap.py` retains the local SOAP implementation, including its attribution to PyTorch AdamW and GaLore. The local SOAP MIT license, copyright 2024 Nikhil Vyas, is retained at [licenses/SOAP.txt](licenses/SOAP.txt).
- `code/cstride/mmnn.py` and the FMMNN model implementation implement the modulated neural-field architecture used by the released checkpoints.

The third-party license files apply to their respective components. Original C-STRIDE code and model weights are distributed under the root MIT license. Released input data are distributed under data/LICENSE.md (CC BY 4.0). The manuscript is linked separately on arXiv and is not included in the public export.
