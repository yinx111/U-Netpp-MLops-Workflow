# Unet++ MLOps for Remote-Sensing Segmentation

This repository provides an MLOps workflow around the project https://github.com/yinx111/UNetpp-Semantic-Segmentation-on-Multispectral-Satellite-Imagery.  
It manages datasets with **DVC** and uses **DVC stages** to automate the end-to-end pipeline, including **training, evaluation, quality gate checks, and model registration**. Training runs and experiment tracking are handled with **MLflow**. **Github actions CI** cover **lint-and-type-checking, smoke tests, and full DVC pipeline execution**. Both DVC and MLflow are integrated and hosted on **DagsHub**.

## Install dependencies

```
conda create --name smp python=3.9 -y 
conda activate smp
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install segmentation-models-pytorch
pip install -r requirements.txt

```

## Set up dvc remote

```
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl {your dagshub repositories url}.s3
dvc remote modify origin --local access_key_id {your_token}
dvc remote modify origin --local secret_access_key {your_token}
```
You can get your DVC remote token from your DagsHub repository page

![alt text](<dvc remote setting ok.png>)

## Configure MLflow

Create a .env file 
```
MLFLOW_TRACKING_URI={ your dagshub repositories url }.mlflow
MLFLOW_TRACKING_USERNAME= { your user name }
MLFLOW_TRACKING_PASSWORD= { your access token }
```
You can get your Personal Access Tokens from your DagsHub setting page

![alt text](<mlflow remote setting.png>)

## Run

```
dev repro --force
```

Monitor the training process using MLflow UI

![alt text](mlflow.png)

## CI
The CI pipeline is triggered in two cases:
·Pushes to the main or master branch.
·Pull requests targeting the main or master branch.

![alt text](CI.png)

## Notes
This project is still under development. In the future, **data preprocessing** will be integrated into the pipeline, and I plan to build a simple web app and add **continuous delivery (CD)** capabilities.