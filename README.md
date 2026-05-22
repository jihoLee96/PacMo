# PACMO Artifact

This repository contains the public artifact for the USENIX Security '26 paper:

**MOTION IN THE CLEAR: Reconstructing VR User Behavior from Network Traffic**

The artifact provides a minimal public release of the PACMO implementation and supporting scripts for artifact availability verification. It includes source code for the main PACMO pipeline, a waveform-profiling example, and downstream inference scripts.

## Repository Structure

```text
.
├── fig1_fig2.ipynb
├── injection_script.py
├── LICENSE
├── phase1_mapping.py
├── phase2_reconstruct.py
├── README.md
├── 5.App5_waveform/
├── phase3_inference/
└── tape/
```

### Components

phase1_mapping.py and injection_script.py: scripts for PACMO Phase I, encoding-agnostic mapping and waveform-based profiling.  
phase2_reconstruct.py: script for PACMO Phase II, motion reconstruction from packet-derived fields.  
5.App5_waveform/: small example waveform/profiling trace and intermediate outputs.
phase3_inference/: downstream behavioral inference scripts, including user-identification and feature-extraction code.  
fig1_fig2.ipynb: supporting analysis notebook for the traffic-measurement figures.
tape/: example controlled-input files used for waveform-style profiling.

### Data Availability

This public artifact does not include raw participant motion traces, raw packet captures from user-study sessions, or raw virtual-typing logs.

VR motion data can serve as biometric behavioral telemetry, and typing traces may contain sensitive input information. Therefore, raw human-subject data and raw user-study packet captures are not publicly released. The included example files are provided only to illustrate the PACMO workflow.

### License
See LICENSE.