# ForestFapar

Random Forest based pipeline for estimating biophysical variables (fAPAR, LAI, etc.) from UAV-derived data.

## Project Structure

```
ForestFapar/
├── estimation.py              # Entry point
├── config/
│   ├── splits_params.py       # Data splitting & normalization config
│   └── RF_params.py           # Random Forest hyperparameters & prediction config
├── processing/
│   ├── splits_processing.py   # Dataset splitting & normalization
│   └── RF_processing.py       # RF training, testing, inference
└── utils/
    ├── metrics.py             # Error metrics (R2, RMSE, MAE, NASH, etc.)
    ├── example_dataset_train_val_test.xlsx   # Training/validation/test dataset
    └── example_dataset_predict.xlsx          # Prediction dataset
```

## Installation

Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate forestfapar
pip install .
```

Option 2: pip
```bash
pip install numpy pandas matplotlib scikit-learn joblib openpyxl
```

## Configuration

### Data splitting (`config/splits_params.py`)

| Parameter | Description | Default |
|---|---|---|
| `dataset_path` | Path to training dataset Excel file | `utils/example_dataset_train_val_test.xlsx` |
| `features` | List of input feature column names | `['logpre','logvpd','chi','logppfd','gtmp','AI','loggpp','soc','logtp','logtn']` |
| `target` | Target variable column name | `'fapar'` |
| `trials` | Query conditions to filter data | `{'all_data': 'index == index'}` |
| `training_size` | Fraction of data for training | `0.7` |
| `test_size` | Fraction of data for testing | `0.15` |
| `normalize` | Whether to apply min-max normalization | `True` |
| `output_path` | Directory for saving split data | `output_splits` |

### Random Forest hyperparameters (`config/RF_params.py`)

| Parameter | Description | Default |
|---|---|---|
| `n_estimators` | Number of trees in the forest | `100` |
| `min_samples_split` | Minimum samples to split a node | `2` |
| `criterion` | Split quality measure | `'squared_error'` |
| `max_features` | Number of features per split | `'sqrt'` |
| `min_samples_leaf` | Minimum samples at a leaf node | `1` |
| `random_state` | Random seed for reproducibility | `42` |

### Prediction config (`config/RF_params.py` - `RF_predict`)

| Parameter | Description | Default |
|---|---|---|
| `dataset` | Path to prediction dataset | `utils/example_dataset_predict.xlsx` |
| `features` | Input features (must match training) | same as above |
| `target` | Target variable name | `'fapar'` |
| `normalize` | Whether to normalize prediction data | `True` |

## Usage

```bash
cd ForestFapar
python estimation.py
```

## Pipeline

1. **Data Splitting** - Loads the training dataset, applies min-max normalization, splits into train/validation/test sets (70/15/15), and saves to `output_splits/`.
2. **RF Training** - Trains a Random Forest regressor, computes feature importance, saves the model and validation metrics.
3. **RF Testing** - Evaluates the trained model on the test set, generates scatter plot and error metrics.
4. **RF Prediction** - Loads the prediction dataset, applies normalization using training min/max values, runs inference, and saves results.

## Outputs

After running, the following outputs are generated:

- `output_splits/` - Normalized train/val/test splits and min-max values
- `trial1_RF/` - Trained model, feature importance plot, validation metrics
- `trial1_RF/test/` - Test scatter plot and metrics
- `trial1_RF/outputs_trial1_RF.xlsx` - Prediction results

## Dependencies

- Python >= 3.8, < 3.12
- numpy >= 1.23
- pandas >= 1.4
- matplotlib >= 3.5
- scikit-learn >= 1.1
- joblib >= 1.1
- openpyxl >= 3.0
