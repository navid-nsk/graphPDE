# GraphPDE: Physics-Informed Graph Neural Network for Population Dynamics

A novel physics-informed deep learning framework that combines Graph Neural Networks with Partial Differential Equations (PDEs) for modeling spatio-temporal population dynamics. GraphPDE learns interpretable reaction-diffusion dynamics on graph-structured data, with applications to demographic forecasting, traffic prediction, and epidemiology.

---

## Data Download

Download the datasets required for training and evaluation:

**[Download Data from Figshare](https://doi.org/10.6084/m9.figshare.31329364)**

This includes:
- **Canada Census Data** - Population dynamics dataset for demographic forecasting
- **METR-LA Traffic Data** - Traffic flow dataset for generalizability experiments

After downloading, extract the data files to the `data/` directory.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Figure Generation](#figure-generation)
- [Generalizability Experiments](#generalizability-experiments)
- [Citation](#citation)

---

## Features

- **Physics-Informed Architecture**: Learns interpretable diffusion coefficients, growth rates, and carrying capacities
- **Graph-Based Spatial Modeling**: Operates on arbitrary graph structures (city networks, road networks, etc.)
- **Multi-Step Forecasting**: Rolling predictions across multiple time horizons (5, 10, 15, 20 years)
- **Attention-Based Interactions**: Multi-head attention for learning inter-group dynamics
- **CUDA Acceleration**: Custom CUDA kernels for efficient graph Laplacian computation
- **Domain Generalization**: Demonstrated on demographics, traffic, and epidemiology domains

---

## Requirements

- Python 3.8+
- PyTorch 1.12+ with CUDA support
- NVIDIA GPU with compute capability 7.5+ (RTX 20xx/30xx/40xx series)
- CUDA Toolkit 11.0+

### Python Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy matplotlib seaborn
pip install scikit-learn tqdm
pip install geopandas shapely  # For figure generation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/graphpde.git
cd graphpde
```

### 2. Install CUDA Extensions (Required for GPU Acceleration)

The custom CUDA kernels provide significant speedup for graph Laplacian operations.

```bash
cd cuda_extensions/graph_laplacian/
python setup.py install
```

**Alternative (Development Mode):**
```bash
pip install -e .
```

**Verify Installation:**
```python
import torch
import graph_laplacian_cuda
print("CUDA extension installed successfully!")
```

**Troubleshooting CUDA Installation:**

| Issue | Solution |
|-------|----------|
| `nvcc not found` | Add CUDA bin to PATH: `export PATH=/usr/local/cuda/bin:$PATH` |
| `No matching GPU architecture` | Update `setup.py` with your GPU's compute capability |
| `cl.exe not found` (Windows) | Install Visual Studio Build Tools with C++ workload |

**Supported GPU Architectures:**
- RTX 20xx series: `compute_75`
- RTX 30xx series: `compute_80`, `compute_86`
- RTX 40xx series: `compute_89`

### 3. Verify Installation

```bash
python -c "from model_graphpde import create_graphpde_model; print('GraphPDE ready!')"
```

---

## Quick Start

### Run All Evaluations (Peer Review)

The master script runs all evaluations and generates figures:

```bash
python run_all_evaluations.py
```

This executes:
1. **Main Model Evaluation** - Single-step prediction on test set
2. **Multi-step Horizon Evaluation** - 5/10/15/20-year forecasts
3. **Traffic Generalizability** - Domain transfer evaluation
4. **Figure Generation** - All publication figures

**Options:**
```bash
python run_all_evaluations.py --skip_figures    # Skip figure generation
python run_all_evaluations.py --only_figures    # Only generate figures
python run_all_evaluations.py --skip_traffic    # Skip traffic evaluation
```

All results are saved to the `results/` folder.

---

## Project Structure

```
graphpde/
├── model_graphpde.py              # Main GraphPDE model architecture
├── trainer_graphpde.py            # Training loop for single-step
├── dataloader_with_city_filter.py # Data loading with city/ethnicity filtering
├── run_training_graphpde_updated.py  # Main training script
├── evaluate_test_graphpde.py      # Test set evaluation
├── run_all_evaluations.py         # Master evaluation script
│
├── cuda_extensions/               # CUDA kernels
│   └── graph_laplacian/
│       ├── setup.py
│       ├── graph_laplacian_cuda.cpp
│       └── graph_laplacian_kernel.cu
│
├── experiments/
│   └── multistep/                 # Multi-step prediction experiments
│       ├── run_training_graphpde_multistep.py
│       ├── trainer_graphpde_multistep.py
│       └── evaluate_multistep_graphpde.py
│
├── generalizability/              # Domain transfer experiments
│   ├── traffic/
│   │   ├── run_training_v2.py
│   │   └── evaluate.py
│   └── epidemiology/
│       ├── run_training_v2.py
│       └── evaluate.py
│
├── analysis/                      # Figure generation scripts
│   ├── generate_all_figures.py
│   ├── compare_all_models_figure_2.py
│   ├── create_city_analysis_figure_3.py
│   └── ...
│
├── baselines/                     # Baseline model implementations
│   ├── mlp_baseline.py
│   ├── gcn_baseline.py
│   └── ...
│
├── data/                          # Data files
│   ├── train_rd.csv
│   ├── val_rd.csv
│   ├── test_rd.csv
│   ├── graph_large_cities_rd.pkl
│   └── da_canada.csv
│
├── results/                       # Output results
└── checkpoints/                   # Model checkpoints
```

---

## Training

### Single-Step Training (Main Model)

Train GraphPDE on population prediction with 5-year horizons:

```bash
python run_training_graphpde_updated.py \
    --epochs 400 \
    --batch_size 1024 \
    --lr 0.0001 \
    --experiment_name my_experiment
```

#### Training on Specific Cities

Filter training to specific Canadian cities:

```bash
# Train on Greater Toronto Area only
python run_training_graphpde_updated.py \
    --city_filter Toronto Mississauga Brampton Markham Vaughan \
    --experiment_name gta_model

# Train on Western Canada
python run_training_graphpde_updated.py \
    --city_filter Vancouver Surrey Burnaby Delta Abbotsford \
    --experiment_name western_model
```

#### Training on All Canadian Cities

Remove the city filter to train on all available cities:

```bash
python run_training_graphpde_updated.py \
    --city_filter ""  # Empty to use all cities
    --experiment_name full_canada_model
```

**Note:** Training on all cities requires more GPU memory and longer training time.

#### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 400 | Number of training epochs |
| `--batch_size` | 1024 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--weight_decay` | 1e-6 | L2 regularization |
| `--grad_clip` | 500.0 | Gradient clipping threshold |
| `--label_smoothing` | 0.1 | Label smoothing factor |
| `--scheduler` | plateau | LR scheduler: `plateau`, `cosine`, `step`, `cosine_warmup` |
| `--patience` | 15 | Early stopping patience |
| `--city_filter` | [10 cities] | Cities to include in training |
| `--resume` | - | Resume from checkpoint |

#### Available Cities

Default training cities (Recommended for Experiment):
- Toronto, Mississauga, Brampton, Markham, Vaughan
- Abbotsford, Winnipeg, Surrey, Delta, Burnaby

Main model trained on cities:
- Richmond Hill, Oakville, Milton, Oshawa, Whitby
- Hamilton, Guelph, Cambridge, Kitchener, Waterloo
- London, Windsor, Barrie, St. Catharines, Kingston
- Thunder Bay, Greater Sudbury / Grand Sudbury
- Vancouver, Richmond, North Vancouver, Coquitlam, Langley
- Chilliwack, Kelowna, Nanaimo, Saanich
- Calgary, Edmonton, Rocky View County
- Regina, Saskatoon
- Montréal, Laval, Longueuil, Brossard, Québec, Gatineau
- Halifax

### Multi-Step Training

Train for rolling multi-step predictions (5, 10, 15, 20-year horizons):

```bash
cd experiments/multistep/

# Basic multi-step training
python run_training_graphpde_multistep.py \
    --epochs 400 \
    --lr 0.001

# With curriculum learning (gradually increase prediction steps)
python run_training_graphpde_multistep.py \
    --use_curriculum \
    --curriculum_warmup_epochs 30

# With custom step loss weights
python run_training_graphpde_multistep.py \
    --step_loss_weights 0.5 0.75 1.0 1.0
```

#### Multi-Step Training Features (NL Framework)

The multi-step trainer includes advanced training techniques:

| Feature | Description |
|---------|-------------|
| **Gradient Spike Protection** | Adaptive clipping based on gradient history |
| **Warmup Scheduler** | Linear warmup + cosine annealing with restarts |
| **Learnable LR Controller** | Hypergradient descent for adaptive LR |
| **Multi-Frequency Updates** | Different update rates per module |
| **Per-Module LR Scaling** | Physics params: 0.01x, Embeddings: 0.5x, MLP: 1.0x |

```bash
# Disable specific features
python run_training_graphpde_multistep.py \
    --no_spike_protection \
    --no_warmup_scheduler \
    --no_learnable_lr
```

---

## Evaluation

### Test Set Evaluation

```bash
python evaluate_test_graphpde.py \
    --checkpoint_dir ./model \
    --output_dir ./results/evaluation
```

### Multi-Step Horizon Evaluation

Evaluate model performance at different forecast horizons:

```bash
cd experiments/multistep/
python evaluate_multistep_graphpde.py \
    --checkpoint_dir ../../checkpoints/graphpde_multistep \
    --split test
```

**Output Format:**
```
================================================================================
MULTI-STEP PREDICTION RESULTS (Averaged over 3 models)
================================================================================
Horizon       Steps               MAE              RMSE           MAPE(%)                    R2
--------------------------------------------------------------------------------------------------
5-year            1    14.61 +/- 0.02    36.37 +/- 1.53    59.24 +/- 0.16    0.7526 +/- 0.0205
10-year           2    18.97 +/- 0.02    46.33 +/- 2.66    50.21 +/- 1.91    0.6959 +/- 0.0341
15-year           3    23.58 +/- 0.39    56.27 +/- 0.18    50.91 +/- 0.94    0.6617 +/- 0.0021
20-year           4    25.98 +/- 0.75    63.45 +/- 8.09    54.59 +/- 1.40    0.5629 +/- 0.1146
--------------------------------------------------------------------------------------------------
```

---

## Figure Generation

### Generate All Figures

```bash
cd analysis/
python generate_all_figures.py
```

### Individual Figure Scripts

#### Figure 2: Model Comparison
```bash
python compare_all_models_figure_2.py
```
Compares GraphPDE against baselines (XGBoost, MLP, GCN, GAT, GraphSage, TGCN, DCRNN).

#### Figure 3: City-Level Analysis
```bash
python create_city_analysis_figure_3.py \
    --checkpoint ../model/best_model.pt
```

#### Figure 5: Toronto Analysis Panels
```bash
# Geographic overview
python panel_a_toronto_overview_figure_5.py

# Turing pattern analysis
python panel_b_turing_zoom_figure_5.py

# Temporal dynamics
python panel_d_dynamics_separated_figure_5.py
```

#### Figure 6: Crystallization Visualization

Visualize settlement patterns for specific ethnicities:

```bash
python create_graphpde_crystallization_figure_6.py \
    --ethnicity China \
    --output-dir ../figures/figure6

# Other ethnicities
python create_graphpde_crystallization_figure_6.py --ethnicity India
python create_graphpde_crystallization_figure_6.py --ethnicity Philippines
```

#### Figure 7: Energy Landscape

```bash
python create_energy_landscape_toronto_chinese_figure_7.py \
    --ethnicity China \
    --output ../figures/figure7/

# For other cities/ethnicities
python create_energy_landscape_toronto_chinese_figure_7.py \
    --ethnicity India
```

### Customizing Figures by City and Ethnicity

Most figure scripts accept city and ethnicity parameters:

```bash
# Analyze specific city
python create_city_analysis_figure_3.py --city Toronto

# Analyze specific ethnicity
python create_graphpde_crystallization_figure_6.py --ethnicity "South Asian"

# Combined analysis
python create_energy_landscape_toronto_chinese_figure_7.py \
    --city Vancouver \
    --ethnicity China
```

**Available Ethnicities:**
- China, India, Philippines, Pakistan, Sri Lanka
- Jamaica, Iran, Korea, Vietnam, Hong Kong
- United Kingdom, United States, Poland, Portugal
- And many more (see data files for complete list)

---

## Generalizability Experiments

GraphPDE architecture generalizes to other spatio-temporal prediction domains.

### Traffic Prediction

Train GraphPDE on traffic flow prediction (METR-LA / PEMS-BAY datasets):

```bash
cd generalizability/traffic/

# Training
python run_training_v2.py \
    --epochs 200 \
    --batch-size 64 \
    --lr 2e-3 \
    --hidden-dim 128 \
    --n-ode-steps 4

# Evaluation
python evaluate.py
```

**Traffic-Specific Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | 128 | Hidden layer dimension |
| `--n-ode-steps` | 4 | ODE integration steps |
| `--ode-method` | euler | ODE solver: `euler` or `rk4` |
| `--no-attention` | - | Disable sensor attention |

### Epidemiology (COVID-19)

Train GraphPDE on disease spread prediction:

```bash
cd generalizability/epidemiology/

# Training
python run_training_v2.py \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dim 64

# Evaluation
python evaluate.py
```

**Epidemiology-Specific Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | 64 | Hidden layer dimension |
| `--n-ode-steps` | 4 | ODE integration steps |
| `--ode-method` | euler | ODE solver |

---

## Model Architecture

GraphPDE combines physics-informed PDEs with graph neural networks:

### Physics Parameters (Learnable, Per Time Period)
- **Diffusion Coefficients**: Spatial spread rates
- **Growth Rates**: Logistic population growth
- **Carrying Capacity**: Maximum population per region
- **Immigration/Emigration Rates**: Population flow

### Neural Components
- **City Embeddings** (512-dim): City-specific dynamics
- **Ethnicity Embeddings** (512-dim): Group-specific patterns
- **Period Embeddings** (64-dim): Temporal variations
- **Census Modulator MLP**: Adjusts physics based on census features
- **Residual MLP**: Learns physics corrections
- **Multi-Head Attention**: Inter-group interactions

### ODE Integration
- RK4 solver with configurable steps
- 5-year integration windows (matching census periods)
- CUDA-accelerated graph Laplacian

---

## Output Files

### Training Outputs
```
checkpoints/<experiment_name>/
├── best_model.pt           # Best validation loss
├── best_r2_model.pt        # Best R² score
├── latest_checkpoint.pt    # Most recent
├── metrics.csv             # Training history
├── physics_parameters.pkl  # Learned physics params
└── test_results.csv        # Test metrics
```

### Evaluation Outputs
```
results/
├── evaluation_horizons_test.csv    # Multi-step results
├── test_metrics.csv                # Single-step metrics
└── figures/                        # Generated figures
```


---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Canadian Census Data
- METR-LA and PEMS-BAY Traffic Datasets
- US COVID-19 County Data
