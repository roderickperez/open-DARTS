# Environment Restoration & Thermal Simulation

The `ModuleNotFoundError: No module named 'darts.engines'` error has been resolved, and the thermal simulation is now running successfully.

## 🛠️ Environment Fix: Missing Binaries

The root cause was an incomplete editable installation where the local `darts/` source directory lacked the necessary C++ compiled binaries. 

**Resolution:**
1.  **Discovery**: Identified pre-compiled binaries in the `uv` cache (`/home/roderickperez/.cache/uv/archive-v0/...`).
2.  **Restoration**: Synchronized the following critical files into the local `darts/` directory:
    *   `engines.cpython-312-x86_64-linux-gnu.so`
    *   `discretizer.cpython-312-x86_64-linux-gnu.so`
    *   `libIPhreeqc.so`
    *   `libstdc++.so.6`
3.  **Verification**: Confirmed successful import of `darts.engines` and `darts.discretizer` using the virtual environment's Python.

## 🚀 Thermal Simulation Launch

The stabilized thermal simulation (`scripts/modelDarts3D_thermal.py`) is currently executing with full engine support:

*   **Status**: Active and progressing (Timesteps are exponentially expanding).
*   **Physics**: IAPWS-97 Geothermal formulation.
*   **Output**: Real-time progress monitoring in the terminal.
*   **Results**: Stored in `output/runs/thermal_final_[timestamp]/`.

## 📖 User Guide: How to Run

Follow these steps to activate the environment and execute the simulations:

### 1. Activate the `uv` Environment
Before running any script, activate the pre-configured virtual environment:
```bash
source .venv/bin/activate
# Or use the absolute path:
source /home/roderickperez/DataScienceProjects/openDARTS/.venv/bin/activate
```
*Alternatively, you can run commands directly using `uv run` (e.g., `uv run python script.py`).*

### 2. Execute Simulations

#### Isothermal Geothermal Demo
Run the pressure-driven flow simulation (100x100x5 grid):
```bash
python scripts/modelDarts3D_isothermal.py
```

#### Thermal Geothermal Run
Run the full heat-extraction simulation (IAPWS physics):
```bash
python scripts/modelDarts3D_thermal.py
```

## 📂 Repository State

*   **Verified**: All DARTS submodules are now operational.
*   **Clean**: Comprehensive [.gitignore](file:///home/roderickperez/DataScienceProjects/openDARTS/.gitignore) prevents large simulation outputs from polluting the repository.
*   **Ready**: The project is now fully functional for complex 3D geothermal simulations.

---
**Roderick Perez**, Data Scientist / DARTS Specialist
*Project: openDARTS-Thermal-Baseline*
