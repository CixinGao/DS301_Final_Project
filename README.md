# DS301_Final_Project
This repo contains the final project for DS301 in NYU 2025 spring

## Project Title:
ResMLP ODE for Time Series Forecasting

## Project Team Member:
Ning Chen, Sijing He, Cixin Gao

## Project Description:
We proposed a hybrid model that combines a Residual Multi-Layer Perceptron (ResMLP) with a Neural Ordinary Differential Equation (ODE) block for time series forecasting. The ResMLP extracts latent features from a 10-day * 5 features input slice, then outputs the hidden state to the ODE block, which evolves this representation over continuous time. This process would help the model transition from historical data to future prediction more smoothly. Our model is evaluated on gold price data and compared with the hybrid model against standard baselines of ResMLP, MLP, LSTM, and Neural ODEs. Results show that the ResMLP+ODE architecture achieves strong performance, especially after hyperparameter tuning.

## Repository and Code Structure:
Foe this repository, we had "ResMLP+ODE_with_ResMLP.ipynb" as the main coding, "goldprice2001.2.1--2024.1.1.xlsx" as the dataset which was created by "milestone2_dataprocessing.py".

* The baseline model that used to compare with ResMLP+ODE:
* ResMLP: "ResMLP+ODE_with_ResMLP.ipynb"
* LSTM: milestone2_LSTM.py
* MLP: "MLP.ipynb"
* NeuralODE: "Neural ODE.ipynb"

The "Hyperparameter Tuning Result.xlsx" is file that shows our ResMLP+ODE hyperparameter tuning result and why we chose such a hyperparameter group as the final model.

## Example commands to execute the code:
Our Python code is designed to run in Google Colab.
To execute it properly, we recommend downloading both the .ipynb notebook and the associated .csv data files and uploading them to Colab.
First, open the notebook in Colab. Then, use the file panel on the left to upload the .csv file under the “Files” tab. Once uploaded, the code should run smoothly with access to the data.

## Results
<img width="1074" alt="截屏2025-05-10 20 26 05" src="https://github.com/user-attachments/assets/a5446e33-f77c-4192-892f-d6fd71029c82" />

