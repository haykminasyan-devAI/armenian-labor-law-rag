#!/bin/bash
#SBATCH --partition=scalar6000q
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=dataproc
#SBATCH --output=logs/rag_%j.out
#SBATCH --error=logs/rag_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "RAG Pipeline with Llama 3-8B on GPU"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Go to project directory
cd /home/hayk.minasyan/Project/NLP_proj

# Activate virtual environment
source venv/bin/activate

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Run RAG pipeline
echo "Starting RAG pipeline..."
python scripts/test_rag.py

echo "=========================================="
echo "Job completed!"
echo "=========================================="
