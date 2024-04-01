# Nuscenes_tutorial
## conda
```
conda create -n nuscenes_tutorial python=3.8

conda activate nuscenes_tutorial
```
## dataset download
```
mkdir -p /data/datasets/nuscenes  # Make the directory to store the nuScenes dataset in.

wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.

tar -xf v1.0-mini.tgz -C /data/datasets/nuscenes  # Uncompress the nuScenes mini split.

pip install nuscenes-devkit &> /dev/null  # Install nuScenes.
```
## git clone new modules
```
git clone https://github.com/byounghun24/Nuscenes_tutorial.git
```
