# PPO Join Order Optimization

A reinforcement learning system that trains a PPO agent to optimize SQL join orders using real PostgreSQL execution.

## Core Objective

- **Train PPO agent** on 1000 queries with real IMDB database
- **Evaluate agent** on 20 test queries vs PostgreSQL default planner
- **Real database execution** with actual IMDB data

## Quick Start

### Prerequisites
1. PostgreSQL running
2. Python packages: `torch`, `numpy`, `psycopg2-binary`

### Run Complete Experiment
```bash
python run_cold_start_experiment.py
```

This will:
1. Connect to existing IMDB database
2. Train PPO agent on 1000 queries with real data
3. Evaluate on 20 separate test queries
4. Save results and trained model

## Core Files

### Training System
- `ColdStartTraining.py` - Training with real IMDB database
- `PPOAgent.py` - PPO reinforcement learning agent
- `StateEncoder.py` - Neural network state encoding
- `JoinTreeState.py` - Join tree state representation
- `JoinOrderEnv.py` - RL environment with real database execution

### Database & Queries
- `IMDBSchema.py` - IMDB schema definition
- `RealDatabaseExecutor.py` - PostgreSQL execution
- `QueryGenerator.py` - Synthetic query generation
- `IMDBSchema.py` - IMDB schema definition

### Evaluation
- `SeparateTestEvaluation.py` - Evaluation on unseen queries
- `run_cold_start_experiment.py` - Main orchestration script

## Output Files

- `best_model_cold_start.pth` - Trained PPO model
- `cold_start_training_results.json` - Training metrics
- `separate_test_evaluation_results.json` - Evaluation results

## Configuration

The system automatically uses your username for PostgreSQL connection:
```python
{
    'host': 'localhost',
    'port': 5432,
    'database': 'imdb',
    'user': 'eliaruhle',  # Your username
    'password': ''
}
```

## Key Features

- **Real Database Only**: Uses actual IMDB data with real PostgreSQL execution
- **Separate Test Set**: Fair evaluation on unseen queries
- **Real PostgreSQL**: Actual execution time measurement
- **IMDB Schema**: Complete movie database structure
- **Comprehensive Analysis**: Pattern and complexity breakdown 