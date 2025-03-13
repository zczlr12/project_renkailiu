# COMP0248 Coursework 1

## Download the Dataset
```bash
curl -L -o ~/data/camvid.zip\
  https://www.kaggle.com/api/v1/datasets/download/carlolepelaars/camvid
cd data && unzip -L camvid.zip && rm camvid.zip && cd ..
```

## Setup Environments
```bash
conda create -n pytorch_GPU python=3.10
conda activate pytorch_GPU
conda install pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 
pip install -r requirements.txt
```

## Data Loading and Preprocessing
```bash
python src/dataloader.py
```

## Training
```bash
python src/train.py
```

## Testing and evaluation
```bash
python src/evaluation.py
```
