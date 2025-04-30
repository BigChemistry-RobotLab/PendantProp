# PendantProp

## Installation

To install the PendantProp packages and its required dependencies, follow these
steps:

1. Navigate to the root directory of this repository.
2. Create a new virtual environment using Python 3.13. You can create the
   environment using conda with `environment.yml` (`conda env create -f
   environment.yml`).
3. Activate the virtual environment.
4. Run `pip install -e .`

To test if this process has worked, run `python run.py`, and go to
[http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser. You should see
the PendantProp interface displayed there.

Please note that the CPU versions of Jax and Numpyro will be installed in this
process. If you wish to use a GPU, please see the [Numpyro installation
instructions](https://github.com/pyro-ppl/numpyro?tab=readme-ov-file#installation).

- TODO: SECRET_KEY

## Usage

TODO: how to set up the hardware, which data files to provide, which script to
run, and how the interface works.

Details about the settings file.

### Designing an Experiment

TODO

### Initialisation

TODO

### Performing an Experiment Run

TODO
