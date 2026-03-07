# Device Setup README

This repository uses a conda-based Open3D and Open3D-ML setup.
No Docker is used in this workflow.

Once you finish the device setup in this document, continue with [`dataset_prep_readme.md`](dataset_prep_readme.md) for dataset labeling, preprocessing, and training dataset preparation.

## Semantic Segmentation Pipeline Overview

The overall workflow in this project is shown below.
It starts with dataset preparation, moves into model selection and training validation, and then loops through raw-data inference and fine-tuning until the model performs well on both labeled and unseen point clouds.

![Semantic model training pipeline](images/Semantic%20Model%20Training%20Pipeline.png)

In brief, the pipeline is:

- prepare the dataset and split it into training, validation, and test sets
- choose a semantic segmentation model: `KPConv`, `RandLA-Net`, or `PointTransformer`
- modify the model config and hyperparameters, then train the selected model
- run inference on trained-data and raw-data validation samples
- if performance is weak, adjust the dataset, configs, class weights, or model hyperparameters and retrain
- once the results are acceptable, finalize the model and continue with deployment or downstream evaluation

For the detailed step-by-step workflow, use these documents after completing the device setup:

- [`dataset_prep_readme.md`](dataset_prep_readme.md)
- [`ModelTraining_README.md`](ModelTraining_README.md)

## Start here

If you are using an `RTX 50xx` series GPU, build Open3D from source first.
Start with:

```bash
git clone https://github.com/isl-org/Open3D.git
cd Open3D
git submodule update --init --recursive

# Only needed for Ubuntu
util/install_deps_ubuntu.sh

mkdir build
cd build
```

After that, continue with the existing conda setup and CMake build steps in this README.

If you are using other GPUs, and especially if you are on Ubuntu `24.04` or below, you can usually follow the official Open3D / Open3D-ML installation documentation instead of rebuilding everything from source:

- https://github.com/isl-org/Open3D-ML
- https://github.com/isl-org/Open3D
- https://www.open3d.org/docs/release/introduction.html

## Local setup this document is based on

- Conda environment: `o3dml_cuda128`
- Python: `3.10`
- PyTorch: `2.7.1+cu128`
- PyTorch CUDA runtime: `12.8`
- PyTorch C++ ABI: `True`
- Open3D package path: `/home/jeremychia/miniconda3/envs/o3dml_cuda128/lib/python3.10/site-packages/open3d`
- Open3D-ML repo path: `/home/jeremychia/Documents/Open3D-ML`
- Main pipeline script: `/home/jeremychia/Documents/Open3D-ML/scripts/run_pipeline.py`

## Current Open3D build flags

The installed Open3D package reports these build settings:

```text
BUILD_TENSORFLOW_OPS=OFF
BUILD_PYTORCH_OPS=ON
BUILD_CUDA_MODULE=ON
BUILD_SYCL_MODULE=OFF
BUILD_AZURE_KINECT=OFF
BUILD_LIBREALSENSE=OFF
BUILD_SHARED_LIBS=OFF
BUILD_GUI=ON
ENABLE_HEADLESS_RENDERING=OFF
BUILD_JUPYTER_EXTENSION=OFF
BUNDLE_OPEN3D_ML=ON
GLIBCXX_USE_CXX11_ABI=ON
CMAKE_BUILD_TYPE=Release
WITH_OPENMP=ON
Pytorch_VERSION=2.7.1+cu128
CUDA_VERSION=12.0
```

## Important compatibility note

PyTorch is installed as `2.7.1+cu128`, which means `torch.version.cuda` is `12.8`.
The current Open3D metadata still reports `CUDA_VERSION=12.0`.

For `open3d.ml.torch` GPU ops, Open3D and PyTorch need matching CUDA versions.
If Open3D is built with CUDA `12.0` and PyTorch is built with CUDA `12.8`, Open3D-ML falls back to CPU ops.

For this project, rebuild Open3D against CUDA `12.8`.

## GPU architecture flags

- RTX 5090: `-DCMAKE_CUDA_ARCHITECTURES=120`
- RTX 4090: `-DCMAKE_CUDA_ARCHITECTURES=89`
- One build for both 5090 and 4090: `-DCMAKE_CUDA_ARCHITECTURES="89;120"`

## 1. Create the conda environment

```bash
conda create -n o3dml_cuda128 python=3.10 -y
conda activate o3dml_cuda128

python -m pip install --upgrade pip setuptools wheel cmake
python -m pip install -r /home/jeremychia/Documents/Open3D-ML/requirements-torch-cuda.txt
```

That installs:

- `torch==2.7.1+cu128`
- `torchvision==0.22.1+cu128`
- `tensorboard`

## 2. Build Open3D from source inside the conda environment

Use the same feature set as the current installed package, but rebuild it against CUDA `12.8`.

### RTX 5090 build

```bash
conda activate o3dml_cuda128

cd /path/to/src
git clone https://github.com/isl-org/Open3D.git
cd Open3D
git submodule update --init --recursive

# Only needed for Ubuntu
util/install_deps_ubuntu.sh

mkdir -p build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_PYTORCH_OPS=ON \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_SYCL_MODULE=OFF \
  -DBUILD_AZURE_KINECT=OFF \
  -DBUILD_LIBREALSENSE=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_GUI=ON \
  -DENABLE_HEADLESS_RENDERING=OFF \
  -DBUILD_JUPYTER_EXTENSION=OFF \
  -DBUNDLE_OPEN3D_ML=ON \
  -DGLIBCXX_USE_CXX11_ABI=ON \
  -DWITH_OPENMP=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  ..

make -j4
make pip-package
python -m pip install lib/python_package/pip_package/open3d*.whl
```

### RTX 4090 build

Use the same command, but replace:

```bash
-DCMAKE_CUDA_ARCHITECTURES=120
```

with:

```bash
-DCMAKE_CUDA_ARCHITECTURES=89
```

### One wheel for both 5090 and 4090

If the same wheel needs to run on both GPU classes, use:

```bash
-DCMAKE_CUDA_ARCHITECTURES="89;120"
```

## 3. Open3D-ML repo usage

For this workflow, Open3D-ML code is used from:

- `/home/jeremychia/Documents/Open3D-ML/scripts/run_pipeline.py`
- `/home/jeremychia/Documents/Open3D-ML/ml3d/configs`

Open3D already bundles the ML namespace because `BUNDLE_OPEN3D_ML=ON`, so the main requirement is that the Open3D package is built correctly for PyTorch `2.7.1+cu128`.

Optional editable install of the standalone Open3D-ML repo:

```bash
conda activate o3dml_cuda128
cd /home/jeremychia/Documents/Open3D-ML
python -m pip install -e .
```

This editable install is optional for the current pipeline because `run_pipeline.py` imports `open3d.ml`.

## 4. Apply the project-specific script and config patches

After Open3D and Open3D-ML are installed, copy this repository's modified files into the active Open3D / Open3D-ML locations.

These patches are required because they contain project-specific changes to:

- `learning_map` and `learning_map_inv`
- `label_to_names`
- dataset preprocessing and dataset loading behavior
- prediction flow in `run_pipeline.py`
- semantic label definitions used by the training configs

### Copy into the Open3D-ML repo

```bash
conda activate o3dml_cuda128

PROJECT_ROOT=/path/to/Semantic-Segmentation-Driven-Robot-Inclusivity--Evaluating-Construction-Site-Accessibility-from-3D-Point-Clouds
OPEN3D_ML_ROOT=/path/to/Open3D-ML

cp "$PROJECT_ROOT/open3d_modified_scripts/run_pipeline.py" \
   "$OPEN3D_ML_ROOT/scripts/run_pipeline.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantic_segmentation.py" \
   "$OPEN3D_ML_ROOT/ml3d/torch/pipelines/semantic_segmentation.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantickitti.py" \
   "$OPEN3D_ML_ROOT/ml3d/datasets/semantickitti.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/s3dis.py" \
   "$OPEN3D_ML_ROOT/ml3d/datasets/s3dis.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantickitti.yml" \
   "$OPEN3D_ML_ROOT/ml3d/datasets/_resources/semantic-kitti.yaml"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantickitti.yml" \
   "$OPEN3D_ML_ROOT/ml3d/datasets/utils/semantic-kitti.yaml"

cp "$PROJECT_ROOT/configs/"*.yml \
   "$OPEN3D_ML_ROOT/ml3d/configs/"
```

### Copy into the installed Open3D package

`run_pipeline.py` imports `open3d.ml`, so the active site-packages copy of Open3D also needs the same dataset and pipeline replacements.

```bash
conda activate o3dml_cuda128

PROJECT_ROOT=/path/to/Semantic-Segmentation-Driven-Robot-Inclusivity--Evaluating-Construction-Site-Accessibility-from-3D-Point-Clouds

O3D_SITE=$(python - <<'PY'
import pathlib
import open3d as o3d
print(pathlib.Path(o3d.__file__).resolve().parent)
PY
)

cp "$PROJECT_ROOT/open3d_modified_scripts/semantic_segmentation.py" \
   "$O3D_SITE/_ml3d/torch/pipelines/semantic_segmentation.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantickitti.py" \
   "$O3D_SITE/_ml3d/datasets/semantickitti.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/s3dis.py" \
   "$O3D_SITE/_ml3d/datasets/s3dis.py"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantickitti.yml" \
   "$O3D_SITE/_ml3d/datasets/_resources/semantic-kitti.yaml"

cp "$PROJECT_ROOT/open3d_modified_scripts/semantickitti.yml" \
   "$O3D_SITE/_ml3d/datasets/utils/semantic-kitti.yaml"
```

## 5. Verify the installation

```bash
conda activate o3dml_cuda128

python - <<'PY'
import torch
import open3d as o3d

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("open3d:", o3d.__version__)
print("open3d cuda:", o3d._build_config["CUDA_VERSION"])
print("open3d pytorch:", o3d._build_config["Pytorch_VERSION"])
print("open3d abi:", o3d._build_config["GLIBCXX_USE_CXX11_ABI"])

import open3d.ml.torch as ml3d
print("open3d.ml.torch import: OK")
PY
```

Expected result:

- `torch.version.cuda` is `12.8`
- `open3d._build_config["CUDA_VERSION"]` is also `12.8`
- `open3d._build_config["Pytorch_VERSION"]` is `2.7.1+cu128`
- `open3d.ml.torch` imports without falling back to CPU

### Check the CPU and GPU torch op files

Also verify that the installed Open3D package contains both the `cpu` and `cuda` backends:

```bash
conda activate o3dml_cuda128

python - <<'PY'
import pathlib
import open3d as o3d

root = pathlib.Path(o3d.__file__).resolve().parent
checks = [
    root / "cpu",
    root / "cuda",
    root / "cpu" / "open3d_torch_ops.so",
    root / "cuda" / "open3d_torch_ops.so",
]

for path in checks:
    print(f"{path}: {'OK' if path.exists() else 'MISSING'}")
PY
```

Notes:

- In some builds, `cpu/open3d_torch_ops.so` is a symlink to `../cuda/open3d_torch_ops.so`. That is normal.
- If the `cpu` or `cuda` directories are missing, or `open3d_torch_ops.so` is missing, the install is incomplete.
- If that happens, download the replacement `cpu` / `cuda` files from the shared folder below and restore them into the installed Open3D package:

```bash
https://drive.google.com/drive/folders/1YZJJGOH4c8iePaBs3cjnRpVS7UWqKNCG?usp=sharing
```

Use the downloaded files to restore the installed Open3D package's `cpu/` and `cuda/` directories.
After restoring them, rerun the verification commands in this README.

## 6. Run Open3D-ML

### Training

```bash
conda activate o3dml_cuda128

python /home/jeremychia/Documents/Open3D-ML/scripts/run_pipeline.py torch \
  -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml \
  --device cuda \
  --device_ids 0 \
  --split train
```

### Prediction

```bash
conda activate o3dml_cuda128

python /home/jeremychia/Documents/Open3D-ML/scripts/run_pipeline.py torch \
  -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml \
  --device cuda \
  --device_ids 0 \
  --split predict \
  --ckpt_path /path/to/ckpt.pth \
  --predict_seq testfullpcd \
  --pred_out /path/to/output
```

## 7. Config assumptions in the current pipeline

The modified config file at `/home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml` already contains:

- `dataset.dataset_path: /home/jeremychia/Documents/Point_clouds/Training/dataset`
- `learning_map_inv`, which is required by the custom `predict` flow
- `model.name: KPFCNN`
- `pipeline.name: SemanticSegmentation`

If the dataset location changes, override it from CLI:

```bash
python /home/jeremychia/Documents/Open3D-ML/scripts/run_pipeline.py torch \
  -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml \
  --dataset.dataset_path /path/to/dataset
```

## 8. Troubleshooting

- If `import open3d` fails with `undefined symbol: ZSTD_decompressStream`, the Open3D install in the conda environment is broken. Rebuild and reinstall the wheel inside the active conda environment instead of copying package files manually.
- If `import open3d` fails during GUI import, the installed wheel is incomplete or mismatched. Rebuild the wheel in the same environment where it will be used.
- If `open3d.ml.torch` warns that Open3D was built with CUDA `12.0` while PyTorch was built with CUDA `12.8`, rebuild Open3D with CUDA `12.8`.
- If this environment only targets one GPU type, use a single CUDA architecture flag (`120` for 5090 or `89` for 4090) to reduce build size and compile time.

## References

- NVIDIA CUDA GPU compute capability table: https://developer.nvidia.com/cuda-gpus
- Open3D build from source documentation: https://www.open3d.org/docs/release/compilation.html
- Open3D-ML installation notes: https://github.com/isl-org/Open3D-ML#installation
