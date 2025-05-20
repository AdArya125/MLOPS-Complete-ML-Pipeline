# MLOPS-Complete-ML-Pipeline
This project covers the end to end understanding for creating an ML pipeline and working around it using DVC for experiment tracking and data versioning(using AWS S3)
<br><br>

## MLOps: Complete ML Pipeline for Spam Classification

This project demonstrates a modular and reproducible machine learning pipeline for SMS spam detection using a Random Forest classifier. The pipeline is implemented with [DVC](https://dvc.org/) for data and experiment versioning, and [DVCLive](https://dvc.org/doc/dvclive) for metrics tracking. It adheres to MLOps principles including modular components, parameterization, version control, and experiment reproducibility.

<br> 

---
<br>

## Objective

Classify SMS messages as **spam** or **not spam** using a machine learning pipeline structured with versioned stages, reproducible experiments, and scalable storage.

<br> 

---
<br>

## Pipeline Stages

All stages are located in the `src/` directory and defined in `dvc.yaml`.

1. **Data Ingestion**  
   → `data_ingestion.py`  
   Loads the raw dataset (`spam.csv`) and saves it to `data/raw/`.

2. **Data Preprocessing**  
   → `data_preprocessing.py`  
   Cleans the text data (lowercasing, removing punctuation, etc.).

3. **Feature Engineering**  
   → `feature_engineering.py`  
   Converts preprocessed text into numerical vectors using TF-IDF or CountVectorizer.

4. **Model Building**  
   → `model_building.py`  
   Trains a Random Forest classifier and saves the model artifact to `models/`.

5. **Model Evaluation**  
   → `model_evaluation.py`  
   Evaluates model performance using Accuracy, Precision, and Recall. Metrics are logged via DVCLive.

<br> 

---
<br>

## Parameters

Parameters for various stages are stored in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.15

feature_engineering:
  max_features: 45

model_building:
  n_estimators: 20
  random_state: 2
```

To access parameters in code:

```python
import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

params = load_params("params.yaml")
```


<br> 

---
<br>


## Reproducing the Pipeline

> Requires: `dvc`, `dvclive`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`

1. **Initialize DVC (only once):**
   ```bash
   dvc init
   ```

2. **Set up pipeline (once):**
   ```bash
   dvc repro
   ```

3. **Check DAG:**
   ```bash
   dvc dag
   ```

4. **Track code & pipeline with Git:**
   ```bash
   git add .
   git commit -m "Setup ML pipeline"
   git push
   ```

<br> 

---
<br>


## Running Experiments

1. **Install DVCLive**
   ```bash
   pip install dvclive
   ```

2. **Run an experiment**
   ```bash
   dvc exp run
   ```

3. **View results**
   ```bash
   dvc exp show
   ```

4. **Compare or apply experiments**
   ```bash
   dvc exp apply <exp-id>
   dvc exp remove <exp-id>
   ```

Each experiment creates a new snapshot of `dvclive/` with metrics and parameter logs.

Example usage in `model_evaluation.py`:
```python
from dvclive import Live

with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy", accuracy)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_params(params)
```

<br>

---
<br>

## Remote Storage (Optional: S3 via DVC)

1. **Install dependencies**
   ```bash
   pip install dvc[s3]
   pip install awscli
   ```

2. **Configure AWS**
   ```bash
   aws configure
   ```

3. **Set remote**
   ```bash
   dvc remote add -d myremote s3://your-bucket-name
   dvc push
   ```

4. **Track with Git**
   ```bash
   git add .dvc/config
   git commit -m "Add S3 remote"
   git push
   ```

<br> 

---
<br>


## Cleanup

Remove AWS resources after use, delete the S3 bucket and IAM user from your AWS console.

---
