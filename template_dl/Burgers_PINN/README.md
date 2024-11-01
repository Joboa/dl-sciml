# Physics-Informed Neural Network for Burger's Equation

This project implements a Physics-Informed Neural Network (PINN) to solve the Burger's equation.

## Project Structure
```
Root/
│
├── Input/
│   └── burgers_shock.mat
│
├── Models/
│   └── model_neuralnet1/
│       ├── model.pt
│       └── training_loss.csv
│
├── Results/
│   └── prediction_results.png
│
├── Scripts/
│   ├── config.py
│   ├── data_utils.py
│   ├── inference.py
│   ├── models.py
│   └── train.py
│
└── README.md
```

## Setup
1. Clone this repository
2. Install required packages:
   ```
   pip install torch numpy scipy matplotlib pyDOE pandas
   ```
3. Ensure the data file `burgers_shock.mat` is in the `Input` folder

## Usage
1. Train the model:
   ```
   python Scripts/train.py
   ```
2. Run inference and generate visualizations:
   ```
   python Scripts/inference.py
   ```

## Results
The model predicts the solution to the Burger's equation. Results are visualized in `Results/prediction_results.png`, showing:
- The predicted solution u(t,x)
- Comparison with exact solution at different time slices
- Training data points used

## Configuration
Adjust hyperparameters in `Scripts/config.py`:
- Data loading parameters (N_u, N_f)
- Training parameters (epochs, learning rate)
- Model architecture
- Viscosity coefficient

## Files Description
- `config.py`: Configuration parameters
- `data_utils.py`: Data loading and preprocessing
- `models.py`: PINN architecture definition
- `train.py`: Training script
- `inference.py`: Prediction and visualization
