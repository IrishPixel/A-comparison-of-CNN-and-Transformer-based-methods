# Preprocessing

Requires `openslide-tools` and a python environment using `requirements.txt`

1. Run `extractTiles-ws.py` with the arguments:

| Variable | Description |
| ----------- | ----------- |
| -s| Path to WSI folder |
| -o | Path to output folder |
| --skipws | Skip tessellation if annotation is missing. Default False|
| -px | Size of image in pixels |
| -um | Size of image in microns |
| --num_threads | Number of threads |
| --augment | Augment extracted tiles with flipping/rotating |
| --ov | Size of the overlapping between 0 and 1 |

2. Run `Normalize.py` with the arguments:

| Variable | Description |
| ------------- | ---------- |
| -ip | Input path of tiles |
| -op | Output path |
| -si | Image path for normalization template |
| -nt | Number of threads |
| -pl | Clini table (if needed) |

# Resnet18

The normalized Tiles should be stored in Data in two directories, Train and Test. For my runs I used a 60/40 train/test split.

Then call `train.py` using the following arguments:
| Variable | Description |
| ---------- | --------- |
| -n | The name of the saved model |
| -m | The torch model being used (e.g torchvision.models.Resnet18) |
| -tr | Path to the training data |
| -ts | Path to the test data |
| -ep | Number of Epochs |
| -bs | Batch size |
| -lr | Learning rate|
| -op | Output path |

Visualizations are included in the `train.py` file.

# GTP

For GTP do not use tiled images as that will be part of the method. Instead skip to the normalization and normalize the WSIs.

### Step 1:

a) Tiling:

Run `src/tile_WSI.py` using the following arguments:

| Variable | Description |
| ---------- | ------------ |
| -s | Tile size |
| -e | Tile overlap from 0 - 1 |
| -j | number of threads |
| -B | Max percentage of background aloud |
| -o | Output path |
| -M | Desired magnification. -1 by default |

b) Feature Extractor:

Modify `config.yaml` in '/feature_extractor' to suit what you are looking for.
Before training the feature extractor put the paths to all of the patches into 'all_patches.csv'.

Run `run.py` 

c) Counstructing Graph:

Run `Build_graphs.py` in '/feature_extractor' using the following arguments:
| Variable | Description |
| --------- | ----------- |
| --weights | Path to the pretrained feature extractor |
| --dataset | Path to patches |
| --output | Output path |

### Step 2:

Train the Transformer using `bash scripts/train.sh` and store the model and logging files under 'graph_transformer/saved_models' and 'graph_transformer/runs' respectively.

To then evaluate the model run `bash scripts/test.sh` 

Split the data into training, validation, and testing datasets and store them in text files like val.txt in data.

### Step 3:

To generate the GraphCAM visualization first generate the model:
`bash scripts/get_graphcam.sh`

To visualize the GraphCAM:
`bash scripts/vis_graphcam.sh`
