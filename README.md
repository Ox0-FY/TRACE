# TRACE: Curriculum-guided Meta-learning Framework for Robust Biosensor Calibration
## Introduction

This project implements a Curriculum Learning + FiLM (Feature-wise Linear Modulation) meta-learning framework designed to address the simulation-to-reality (Sim2Real) gap in biosensor calibration.

Key ideas:

Curriculum learning gradually introduces tasks to improve model robustness

FiLM layers adaptively modulate task-specific feature representations

Achieves high-precision prediction with few-shot real samples

## Project Structure
TRACE/
│── src/
│   │── train.py        # Training logic (main_V2_Curriculum_FiLM)
│   │── main.py         # Entry point that calls the training function
│   │── models.py       # Model definitions (TaskEncoder, FiLMRegressor, FiLMGenerator)
│   │── utils.py        # Data processing and utility functions
│   │── config.py       # Configuration file (paths, hyperparameters, etc.)
│── data/
│   │── real_main.csv   # Real-world training data
│   │── simulated.csv   # Simulated task data
│   │── real_test.csv   # Independent test dataset
│── README.md
