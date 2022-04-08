![Code Formatting](https://github.com/CHIMEFRB/beam-model/workflows/Code%20Formatting/badge.svg?branch=master)
![Functional Tests](https://github.com/CHIMEFRB/beam-model/workflows/Functional%20Tests/badge.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/CHIMEFRB/beam-model/badge.svg?branch=master&t=Qq3KyN)](https://coveralls.io/github/CHIMEFRB/beam-model?branch=master)

# CHIME/FRB BeamModel

Models describing the primary and formed beams for for Canadian Hydrogen Intensity Mapping Experiment (CHIME).


## Installation

The git package can be installed with the following command:

```
pip install git+ssh://git@github.com/CHIMEFRB/beam-model.git
```

If the ssh method does not work, the https method can be used instead:

```
pip install git+https://github.com/CHIMEFRB/beam-model.git
```

To use the data driven beam model, data for the beam must be downloaded by running the script `/bm_data/get_data.py` after installation.

## Developer

In order to develop and contribute code to `chimefrb/beam-model` follow the following steps:

  1. Clone the repository on your development node
     ```
     git clone git@github.com:CHIMEFRB/beam-model.git
     ```
  2. Install development dependencies
     ```
     poetry install
     poetry run get-data
     ```
  3. Create a feature branch with the proposed changes.
  
  4. Run tests
     ```
     poetry run pytest
     ```


