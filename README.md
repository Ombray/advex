## Instructions

### Setting up

The first time you use this repo, run this to get all relevant packages:
```
pipenv install --ignore-pipfile
```

Before running any experiment below, first run
```
pipenv shell
```
to activate the proper environment.

### Running experiments

Run adversarial train/test using a set of $\varepsilon$ values (for adversarial training),
or set of $\lambda$ values (for L1 reg):

```
env PYTHONPATH=. python src/run-grid.py  --hparams=hparams/paper/mushroom-l1-adam.yaml
```

All args for the run are specified in a YAML file passed in as the
`--hparams` argument. Look in the `hparams/paper` directory for a variety of args
related to synthetic or UCI datasets. Results will be produced in a location
under the results directory, with a name matching the hparams file name above, e.g.
`mushroom-l1-adam`. This results directory will be used for the next two
plotting runs.

### Generating plots

- Generate side-by-side bar-plot of weights from two specific settings of the above runs,
identified by the `rows` arg in the YAML file:

```
env PYTHONPATH=. python src/plot-bars.py  --params=plot-params/paper/weights-bars-mushroom.yaml      --file=results/mushroom
```

**Note**: the `--file` arg points to where to look for the results to
  generate the plots. It should be a `pkl` file. If the full file path is not specified, it
  automatically picks the *latest* file under the directory.

- Generate a plot showing how AUC, and a concentration metric changes, with different
values of $\varepsilon$ or $\lambda$, from the above `run-grid` output.

```
env PYTHONPATH=. python src/plot-vars.py  --params=plot-params/paper/wts-l1-mushroom-adam.yaml --file=results/mushroom-l1-adam
```

As in the above plot command, the latest pkl file is automatically
picked if the `--file` path is not fully specified.
