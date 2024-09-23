# Facial Expression Recognition - CNN

Convolutional Neural Network model to recognize facial expressions.

Trained and validated on [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

For model training `train/` and (`test/` or `validation/`) directories in `data/`

## Install requirements to python environment:

```bash
pip install -r requirements.txt
```

## Set PYTHONPATH if imported modules cannot be found:

```bash
export PYTHONPATH=.
```

## Configuration

Edit `config.yaml`

## Train the model:

- Make sure labels are configured correctly in `config.yaml`. They should match directory names

- To train the model and save it to a specific path, run the following command:

```bash
python src/train/main.py --model <PATH_TO_MODEL> --dataset <PATH_TO_DATASET>
# e.g.
python src/train/main.py --model data/models/ExpressionNet_fer --dataset data/datasets/fer2013
```

> [!NOTE]
> Model class name must be in model filename

## Evaluate model on custom images:

```bash
python src/train/main.py --validate --model <PATH_TO_MODEL> --folder <PATH_TO_IMAGES>
# e.g.
python src/train/main.py --validate --model data/models/ExpressionNet_fer --folder data/validation_set
```

## Run expression-recognition client on webcam

To run a webcam facial expression recognition don't specify `--cam`.

This will also run a API client which will average out the predictions over specified amount of time

```bash
python src/client/main.py --model <PATH_TO_MODEL>
# e.g.
python src/client/main.py --model data/models/ExpressionNet_fer
```

## Run API server

API server will run on port 8000

```bash
python src/api/main.py
```

To access API docs: http://localhost:8000/docs

## Run tests

```bash
scripts/tests.sh
```

## Changing requirements/dependencies

- Specified in pyproject.toml
Compiling requirements:

```bash
scripts/requirements.sh
```

## More info

```bash
python src/train/main.py --help
python src/client/main.py --help
```
