# MAFLD Early Detection from Breath Sensor Data using AI by Samuel Bernard and Stephanie Sui

This repository contains AI models and data processing pipelines designed to analyze gas sensor data from breath samples to detect early signs of **Metabolic Associated Fatty Liver Disease (MAFLD)**. The system leverages deep learning techniques (LSTM) to classify individual breathing patterns based on time-series sensor input.

---

## Project Overview

Breath contains trace amounts of volatile organic compounds (VOCs), which can reflect metabolic conditions in the body. By analyzing this data using AI, we aim to:

- Detect early biomarkers of MAFLD
- Enable non-invasive diagnosis from breath data
- Explore the classification of gas profiles using AI models

---

## Repository Structure

```bash
├── data/                  # This folder contains all data about the analysis
|   ├── external/
|   ├── interim/
|   ├── processed/
|   ├── raw/
|   └── test/              # Testing data similar to raw
├── src/
│   ├── models/            # PyTorch-based LSTM classifier and other models
│   ├── data/              # Raw and processed sensor files, normalization.json file
│   ├── utils/             # Data loading, preprocessing, visualization
│   └── scripts/
├── config.yaml            # Training and data configuration
├── requirements.txt       # All required Python packages
├── .gitignore             # Ignoring logs, W&B artifacts, etc.
└── README.md              # You are here!

```
---

## Dataset

- **Input**: Excel files containing time-series readings from gas sensors.
- **Sensors include**:
  - VOCs PID, Methane, Hydrogen Sulfide, Hydrogen, Nitric Oxide, Carbon Dioxide
  - Temperature and Humidity
- **Labels**: Determined by folder name (`Brush/` or `Fasted/`)

---

## Models

- **Type**: Long Short-Term Memory (LSTM)
- **Architecture**:
  - Supports variable-length sequences
  - Configurable hidden size, dropout, and layers
  - Outputs class probabilities via softmax

---


## How to Load and Run the Project in Visual Studio Code (VS Code)
First, you will need to have git-bash installed or open to clone the repository into the folder you want.

### Clone and Open

```bash
git clone 
cd 
code .
```

Or manually: VS Code → `File` → `Open Folder...` → select the repo folder.

### Set Up Python Environment
Then, you will need to set up a virtual environment on VS Code to download the proper packages located in the requirements.txt file.
Below is the protocol to create a virtual environment. You can also create one within the VS Code app.
I would recommend that you open the repo in VS Code, then go to 'Terminal' → 'New Terminal' and write this code:

```bash
python -m venv .venv
.venv\Scripts\activate        # Only if on Windows
source .venv/bin/activate     # Only if on macOS/Linux
pip install -r requirements.txt # This will install the packages you need to run the code
```

- Then in VS Code, open the command palette (`Ctrl+Shift+P`)
- Select: `Python: Select Interpreter` → choose `.venv`

- You also need to install PyTorch to create/run AI models:
- If you have CUDA installed, you can try the CUDA version and follow the steps on the website: https://pytorch.org/get-started/locally/
- If not, I would recommend using the CPU version, and run this in the VS Code terminal:

```bash
pip3 install torch torchvision torchaudio
```
---
## Log in to Wandb
We are currently using wandb to visualize the training progression and validation.
Therefore, you will use my wandb account. I will give you my API key so that you may sign up.
When running the model to train, it will prompt you to enter your api key, that is when you need to paste it.

## Training the current model

```bash
python train_model.py
```

This will:
- Normalize and preprocess your dataset
- Train and validate the current model
- Track metrics in the console and optionally on [Weights & Biases](https://wandb.ai/)

Model checkpoints are saved to:
```yaml
train:
  model_save_path: src/models/checkpoints/lstm_model.pth
```

---

## Running Inference

To classify new breath data stored in `src/data/test`, run:

```bash
python inference.py
```

The script will:
- Load saved normalization stats
- Preprocess the new file
- Output prediction and class probability

---
## Trial Code

If you want to simply try code out and test something, use the sandbox.py file to put your code and run it.
