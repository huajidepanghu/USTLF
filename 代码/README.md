# Reproducible Code for Ultra-short-term Load Forecasting

## 1. Environment

The experiments were conducted with:

```text
Python 3.11
PyTorch 2.5.1
CUDA 12.1
```

Install dependencies:

```bash
pip install torch\=\=2.5.1 --index-url https\://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Data

Place the raw Elia data file in:

```
data/ods003.csv
```

The expected fields are:

- Datetime
- Resolution code
- Elia Grid Load

## 3. File Structure

```
code/
├── data/
│   `-`` ods003.csv
├── outputs/
├── 01_load_data.py
├── 02_preprocess_split_make_windows.py
├── 03_rolling_iceemdan_decomposition.py
├── 04_imf_feature_kmeans_grouping.py
├── models.py
├── 05_train_evaluate.py
├── requirements.txt
`-`` README.md
```

## 4. Main Settings

- Input window length `L \= 96`
- Forecasting horizons `H \= 1, 4, 8, 12`
- Rolling decomposition window `W_dec \= 288`
- ICEEMDAN ensemble size \= 100
- ICEEMDAN noise amplitude \= 0.2
- K-means cluster number `M \= 3`
- Batch size \= 64
- Learning rate \= 0.001
- Max epochs \= 100
- Early stopping patience \= 15
- Random seed \= 42

## 5. Running Steps

### Step 1: Load raw Elia data

```bash
python 01_load_data.py \\
  --input data/ods003.csv \\
  --output outputs/elia_load_raw_standard.csv
```

### Step 2: Preprocess, split, normalize, and build windows

```bash
python 02_preprocess_split_make_windows.py \\
  --input outputs/elia_load_raw_standard.csv \\
  --output_dir outputs \\
  --input_len 96 \\
  --horizons 1 4 8 12
```

### Step 3: Rolling causal ICEEMDAN decomposition

```bash
python 03_rolling_iceemdan_decomposition.py \\
  --input outputs/elia_full_processed.csv \\
  --output_dir outputs/iceemdan_rolling \\
  --decomposition_window 288 \\
  --input_len 96 \\
  --horizon 12 \\
  --ensemble_size 100 \\
  --noise_width 0.2 \\
  --random_seed 42 \\
  --num_workers 8 \\
  --split all
```

### Step 4: Extract IMF features and perform K-means grouping

```bash
python 04_imf_feature_kmeans_grouping.py \\
  --input_dir outputs/iceemdan_rolling \\
  --output_dir outputs/imf_grouping \\
  --n_clusters 3 \\
  --random_seed 42
```

### Step 5: Train and evaluate the model

```bash
python 05_train_evaluate.py \\
  --grouped_dir outputs/imf_grouping \\
  --windows_dir outputs/windows \\
  --normalization_params outputs/normalization_params.json \\
  --output_dir outputs/training \\
  --model_name proposed \\
  --horizon 12 \\
  --input_len 96 \\
  --batch_size 64 \\
  --learning_rate 0.001 \\
  --max_epochs 100 \\
  --early_stopping 15 \\
  --random_seed 42 \\
  --device cuda
```

## 6. Outputs

Main output files include:

- `outputs/elia_full_processed.csv`
- `outputs/windows/`
- `outputs/iceemdan_rolling/`
- `outputs/imf_grouping/`
- `outputs/training/H12/proposed/test_metrics.csv`
- `outputs/training/H12/proposed/test_predictions.csv`

The final metrics include:

- MAE
- RMSE
- MAPE
- R²
