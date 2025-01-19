# Preprocessing

Requires `openslide-tools` and a python environment using `requirements.txt`

1. Run the tiling operations from the GTP method `src/tile_WSI.py`

Run `src/tile_WSI.py` using the following arguments:

| Variable | Description |
| ---------- | ------------ |
| -s | Tile size |
| -e | Tile overlap from 0 - 1 |
| -j | number of threads |
| -B | Max percentage of background aloud |
| -o | Output path |
| -M | Desired magnification. -1 by default |

Exapmle:

`python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <data/Test> "data/*.dzi"`

2. Run `Normalize.py` with the arguments:

| Variable | Description |
| ------------- | ---------- |
| -ip | Input path of tiles |
| -op | Output path |
| -si | Image path for normalization template |
| -nt | Number of threads |
| -pl | Clini table (if needed) |

Example:

`python Normalize.py -ip "path" -op "path" -si normalization_template.jpg`

# Resnet18

The normalized Tiles should be stored in Data in three directories, Train, Val and Test. For my runs I used a 40/20/40 train/validation/test split.

### a) Training:

Then call `train.py` using the following arguments:

| Variable | Description |
| ---------- | --------- |
| -n | The name of the saved model |
| -m | The torch model being used (e.g torchvision.models.Resnet18) |
| -tr | Path to the training data |
| -ts | Path to the test data |
| -ep | Number of Epochs |
| -t | Number of Threads |
| -bs | Batch size |
| -lr | Learning rate|
| -cl | Number of Classes |
| -op | Output path |

Example:

`python train.py -n Resnet18 -m torchvision.models.resnet18 -tr "data path" -ts "data path" -t 8 -ep 50 -bs 8 -lr 0.00018 -cl 4 -op "path"`

### b) Evaluation:

For evaluation call `eval.py` using the following arguments:

| Variable | Description |
| --------- | ----------- |
| -m | The path of the trained model |
| -ts | The path to the test/evaluation set of data |
| -bs | Batch size |
| -t | Number of Threads|
| -l | The path of the outputted log (including the name of the log file) |

This method will return all results in a log file format for later visualisation at your discresion.

# GTP

For GTP do not use tiled images as that will be part of the method. Instead skip to the normalization and normalize the WSIs.

### Step 1:

a) Feature Extractor:

Modify `config.yaml` in '/feature_extractor' to suit what you are looking for.
Before training the feature extractor put the paths to all of the patches into 'all_patches.csv'.

Run `run.py` 

b) Counstructing Graph:

Run `Build_graphs.py` in '/feature_extractor' using the following arguments:
| Variable | Description |
| --------- | ----------- |
| --weights | Path to the pretrained feature extractor |
| --dataset | Path to patches |
| --output | Output path |

Example:

`python build_graphs.py --weights "model path" --dataset "data path" --output "output path"`

### Step 2:

Train the Transformer using `bash scripts/train.sh` and store the model and logging files under 'graph_transformer/saved_models' and 'graph_transformer/runs' respectively.

To then evaluate the model run `bash scripts/test.sh` 

Split the data into training, validation, and testing datasets and store them in text files like val.txt in data.

### Step 3:

To generate the GraphCAM visualization first generate the model:
`bash scripts/get_graphcam.sh`

To visualize the GraphCAM:
`bash scripts/vis_graphcam.sh`
