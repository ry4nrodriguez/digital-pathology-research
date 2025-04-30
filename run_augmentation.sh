#!/bin/bash -l

#$ -P rise2019
#$ -N tissue_aug
#$ -l h_rt=24:00:00
#$ -l gpu=1
#$ -l cpu=4
#$ -j y
#$ -q li-rbsp-gpu
#$ -o augmentation_output.log
#$ -e augmentation_error.log

# Load required modules
module purge
module load python3/3.10.12
module load pytorch/1.13.1

# Print debug information
echo "Starting job at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Install/upgrade any needed packages
pip install --user tqdm opencv-python matplotlib

# Navigate to working directory
cd /projectnb/rise2019/ryanrod_augmented_tissue

# List contents to verify
echo "Contents of working directory:"
ls -l

# Run the augmentation script with detailed output
python -u augment_tissue.py
