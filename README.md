# Exploring World Models for Industrial Systems

Exploratory experiments investigating how learned dynamics models and simulated physical systems could help reason about degradation, future states, and maintenance decisions.

This repository contains two related projects:

1. **Engine Degradation Modelling** using the NASA C-MAPSS turbofan sensor dataset.
2. **Robot Joint Wear Simulation and Forecasting** using a synthetic environment with simplified, hand-designed torque and wear rules.

The goal is to learn by implementing models, inspecting their behaviour, and identifying what would need to be validated before applying these ideas to real industrial systems.

## Projects

### 1. Engine Degradation Modelling

**Directory:** `01-nasa-engine-model/`

This experiment uses the NASA C-MAPSS FD001 turbofan dataset to explore sequence modelling for engine degradation and remaining useful life (RUL) estimation.

The notebook includes:

- Sensor preprocessing and removal of near-constant sensors.
- Sliding-window sequences built from historical sensor readings.
- A PyTorch model with an encoder, LSTM, latent dynamics prediction head, reconstruction decoder, and RUL prediction head.
- A training objective combining reconstruction, latent-dynamics, and RUL losses.
- Evaluation using RUL RMSE and the NASA asymmetric scoring function.
- An exploratory stochastic rollout using dropout to inspect variation in predictions.

**Current evaluation:** The recorded test RUL RMSE is **13.60 cycles**, with a NASA score of **254.11**.

These values are initial results, not evidence of superiority over other methods. The notebook still needs comparisons against simple baselines, stronger validation, and direct evaluation of multi-step future-state predictions. The uncertainty visualisation should also be treated as exploratory rather than as validated sensor-level confidence intervals.

**Data:** The experiment uses NASA C-MAPSS FD001 data. Follow the dataset's applicable access and usage terms; the raw data is not required to be committed to this repository.

### 2. Robot Joint Wear Simulation and Forecasting

**Directory:** `02-robot-wear-model/`

This experiment builds a simplified simulation of a seven-joint robotic arm. It generates episodes with different lifting heights and weights, calculates approximate joint torques, updates simulated wear, and records lift-test outcomes.

The modelling notebook then explores forecasting wear progression from sequences of wear state and stress features.

The experiment includes:

- A synthetic dataset generator with seven joint states.
- Hand-designed torque and wear-degradation rules.
- Variable workload scenarios in the updated simulator.
- An LSTM-based probabilistic model that predicts wear increments.
- A mean prediction and predicted scale, with Monte Carlo dropout used to explore model variability.
- Exploratory future rollouts and light-duty, normal-duty, and heavy-duty scenario comparisons.
- A maintenance-threshold analysis based on simulated wear.

**Important limitation:** The simulator's physical assumptions and wear equations are hand-designed. The resulting dataset is synthetic, so the model can learn patterns created by those rules without necessarily learning the behaviour of a real robot. The model's predictive accuracy, uncertainty calibration, and long-horizon forecasts need to be evaluated quantitatively before drawing operational conclusions.
