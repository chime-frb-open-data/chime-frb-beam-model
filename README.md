# CHIME/FRB Beam Model

Models describing the primary and formed beams for for Canadian Hydrogen Intensity Mapping Experiment FRB backend (CHIME/FRB).


## Installation

The package can be installed from PyPI using the following command:

```
pip install cfbm
```

To use the primary beam model, data for the beam must be downloaded after installation by: 

* Either running the script [cfbm/bm_data/get_data.py](https://github.com/chime-frb-open-data/chime-frb-beam-model/blob/main/cfbm/bm_data/get_data.py).

* Or from within python:
```
from cfbm.bm_data import get_data
get_data.main()
```

## Documentation

Check out the user documentation, [here](https://chime-frb-open-data.github.io/)


