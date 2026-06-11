# PACMO Artifact

This repository contains the public artifact for the USENIX Security '26 paper:

**MOTION IN THE CLEAR: Reconstructing VR User Behavior from Network Traffic**

This repository provides an availability artifact for PACMO. It includes a minimal public release of the PACMO implementation and supporting scripts for artifact availability verification, including source code for the main PACMO pipeline, a waveform-profiling example, and downstream inference scripts.

## Repository Structure

```text
.
|-- fig1_fig2.ipynb
|-- injection_script.py
|-- LICENSE
|-- phase1_mapping.py
|-- phase2_reconstruct.py
|-- README.md
|-- 5.App5_waveform/
|-- phase3_inference/
`-- tape/
```

## Components

`phase1_mapping.py` and `injection_script.py`: scripts for PACMO Phase I, encoding-agnostic mapping and waveform-based profiling.

`phase2_reconstruct.py`: script for PACMO Phase II, motion reconstruction from packet-derived fields.

`5.App5_waveform/`: small example waveform/profiling trace and intermediate outputs.

`phase3_inference/`: downstream behavioral inference scripts, including user-identification and feature-extraction code.

`fig1_fig2.ipynb`: supporting analysis notebook for the traffic-measurement figures.

`tape/`: example controlled-input files used for waveform-style profiling.

## Setup

The scripts were developed for Python 3. Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn tqdm joblib networkx lightgbm torch
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The main Phase I and Phase II scripts require only the Python standard library and `numpy`. The Phase III inference scripts require the additional machine learning packages listed above. The JavaScript feature-extraction helpers under `phase3_inference/identification/featurization/` require Node.js. Running the supporting notebook requires a Jupyter environment.

## Quick Availability Checks

After installing the Python dependencies, a lightweight syntax check can be run with:

```bash
python -m compileall .
```

The included example files in `5.App5_waveform/` and `tape/` are provided to illustrate the PACMO workflow and repository layout. They are not a replacement for the private user-study data used in the paper.

## Data Availability

This public artifact does not include raw participant motion traces, raw packet captures from user-study sessions, or raw virtual-typing logs.

VR motion data can serve as biometric behavioral telemetry, and typing traces may contain sensitive input information. Therefore, raw human-subject data and raw user-study packet captures are not publicly released. The included example files are provided only to illustrate the PACMO workflow.

## License

See `LICENSE`.
