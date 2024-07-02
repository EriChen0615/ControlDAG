# ControlDAG
Code for NAACL 2024 paper: Control-DAG: Constrained Decoding for Non-Autoregressive Directed Acyclic T5 using Weighted Finite State Automata. Paper: https://arxiv.org/abs/2404.06854. Blog: https://www.jinghong-chen.net/3-minute-pitch/. 

# News
[**July 2024**] Training & Inference Code checkpoint released.

# Setup Environment

```bash
# Build Conda Environment
conda create -n dag python=3.9
pip install -r requirements.txt

# Install BLEURT
cd third_party/bleurt
pip install .
cd ../..

# Activate environment
source scripts/activate_env.sh # I've set HF_HOME to `cache/`. You should modify if necessary.
```

# Replication

We provide ready-to-run inference scripts and checkpoints for replicating the experiments on the Schema Guided Dialogue (SGD) dataset. We also provide training scripts to train the AR baseline model and the NAR DA-T5 model from scratch.

## Running Inference on the SGD

You can obtain the checkpoints on request to jc2124 at cam.ac.uk. To replicate results, the following scripts after obtaining the checkpoints. Alternatively, you can train DA-T5 models using the scripts provided. 

```bash
source scripts/activate_env.sh
bash scripts/SGD/test/test-AR-t5.sh # AR baseline (greedy, beam search)
bash scripts/SGD/test/test-baselines.sh # NAR baselines (greedy, beam search, WFSA shortest path)
bash scripts/SGD/test/dag_constrained_beam.sh # CBS-DAG
bash scripts/SGD/test/test-combined_controlled_decoding.sh # Control-DAG with lexical, vocabulary, and length constraints.
```
You may configure decoding hyper-parameters in respective scripts. The variable/argument names are self-explanatory.

> I've also provided scripts for running individual constraints. Their names are self-exploratory under the `scripts/SGD/test` folder. NOTE: you will need to change the experiment folder variable to appropriate path. This is a good place to start understanding the codebase.

**Inference results will be saved under the experiment folder**, for example, `experiments/replication/SGD-DA-T5-SGstar-upsample=5-use_glat=True-lr=1e-4-b=8-ep=40_V0/test/2308-ControlledDecoding/p=3-k=3-FEMIT=True(FTR=0.3)-HLC=True-VC=True(dyn=True-vocab=90-v6)-LC=dfs_memo(USE=regressor-RANK=min_norm_dist_with_penalty-PRUNE_P=0.7-A=1)-ep=39`. The metrics are saved under the `merged-test-evaluation/eval_recorder-stats_log.json` file.

## Training on SGD

```bash
source scripts/activate_env.sh
# Train AR-T5 model
bash scripts/SGD/train/train_AR-T5.sh
# Train DA-T5 model
bash scripts/SGD/train/train_DA-T5.sh
```

# Development

This codebase is based on (a frozen copy of) the `runway_for_ml` framework which I co-authored. To learn more about general runway project structure, please see https://github.com/EriChen0615/runway_for_ml. In short, `runway_for_ml` is a thin wrapper around `pytorch-lightning` that delivers a managed experimental framework for ML researchers and engineers. 

# Issues

Please raise issues in this github repository.