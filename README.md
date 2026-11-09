# Drifting generative modeling for MNIST

## Overview

- `cnn.py`: the CNN classifier
- `drift.py`: the generator
- `jax_utils.py`, `mnist.py`: some utilities
- `cae.py`: a bonus convolutional auto encoder. not better than the classifier.
- `gen_gui.py`, `cnn_gui.py`: test UIs
- `measure_defects.py`, `make_samples.py`, `classifier_eval.py`: for evaluation

## Reproducing

Train the models:

```sh
./main.py train-cnn --epochs 5
./main.py train-drift --encoder cnn --steps 15000
```

Look at the GUI to get some samples:

``` sh
./main.py drift-gui
```

To performs evaluations, run `make_samples.py`, then `measure_defects.py` and `classifier_eval.py`.
