#  PEACH

## Installation
We use `uv` as our environment manager. To set up the environment, run:

```bash
# Create a virtual environment (recommended)
uv venv --python 3.12 .venv
uv sync
```
For developing access, run:
```bash
uv pip install -e .
```

### Install PSTNet CUDA modules
```
cd src/pc_mango/network/util/pstnet/modules
python setup.py install
```



More info:
- activate the venv first so Torch is found
- load nvcc (on cluster usually `module avail cuda` to see the versions, then `module load cuda/<version>`)
- you are usually on a login node so export the torch cuda arch list `export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"`
- `uv pip install -e .` rebuild just in case
- Then try to install the setup script


### Install torch-cluster and torch-scatter

For the pretrained encoder, we need torch-cluster. Install it via:

```bash
uv pip install --no-index --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html torch-scatter
uv pip install --no-index --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html torch-cluster
```
Depending on the cuda version, select a different one, for example
```bash
```bash
uv pip install --no-index --find-links https://data.pyg.org/whl/torch-2.8.0+cu126.html torch-scatter
uv pip install --no-index --find-links https://data.pyg.org/whl/torch-2.8.0+cu126.html torch-cluster
```

---
# Project Structure

## Dataset
- Most likely only using ml_datasets (meta-learning datasets)
- in __init__.py: `get_dataset` function to get dataset by name
- Dataset uses a single .hdf5 file per dataset
- All tasks have multiple trials (i.e. different initial conditions)
- Usually there is a generated pointcloud from the mesh inside the dataset file
- Uses Data modalities config and pc_preprocessing from algorithm config to prepare the data
- Examples:
    - Deformable Block
    - Sheet Deformation
    - Sphere Cloth

## Algorithm
- Lightning module. General task description what to solve
- Examples:
  - PcToMatProp: Pointcloud to Material Prediction, i.e. no simulation
  - ...
- in the `algorithm` config:
  - 1 config file per explicit configuration of an algorithm, for example `pstnet_mlp.yaml` which uses a PSTNet encoder and an MLP decoder (only for the material prediction)
  - Defines the used networks, optimizer, data_modalities (what input is used, e.g. pointcloud, mesh, etc.), and if pc is used: pc_preprocessing
  
## Network
- all neural networks are defined here
- Examples:
  - PSTNet encoder
  - MLP decoder (for material prediction)
  - MaNGO decoder (for simulation)
- The algorithm config defines what networks are used by loading the network config files in its defaults list: `/network/mat_prop_head@mat_prop_head: mlp`
  - This means that in the `mat_prop_head` subconfig, the `mlp.yaml` config file from `/network/mat_prop_head` is loaded
  - This syntax allows to load multiple networks of the same type with different hyperparameters
- Each network has its own config file in `/network`
- When you want to add a new network, you have to:
  - Create the network code in `src/pc_mango/network/<correct_subfolder>`
  - Per hyperparameter config of this network: create a config file in `config/network/<correct_subfolder>` for the network, giving it a name and setting default hyperparameters
  - In the `__init__.py` of the correct subfolder, update the "get..." function to include the new network, use the name given in the config file
---

# Visualizations
- When appropriate, depending on the algorithm and the config of the visualization (in `algorithm/evaluator`), visualizations are created during validation and testing
- Visualizations are saved in the experiment folder under `visualizations/`
- You open the visualizations with [Paraview](https://www.paraview.org/)
- In general, the mesh-based visualzations are saved as one initial geometry and then corresponding offsets for each timestep
- So, you need to apply a "Warp by Vector" filter and select the displacement vector to see the deformed geometry over time
- You can also visualize the pointcloud-based input data.
---

