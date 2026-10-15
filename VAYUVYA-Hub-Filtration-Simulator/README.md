# VAYUVYA Hub Filtration Simulator

A Python + Streamlit prototype for exploring airflow and performance estimates in the VAYUVYA multi-stage air-filter hub.

## Features

- 2-D Darcy-style airflow field through a pre-filter, pleated particle filter, activated carbon layer, and blower throat
- Optional wall-seal leakage and estimated unfiltered bypass
- Blower and system curves with an operating-point estimate
- Pressure drop by layer and airflow visualization
- Size-dependent filtration-efficiency estimates
- Clean-air delivery rate (CADR) and effective air changes per hour (ACH)
- Indoor particulate-matter decay estimate
- Blower power and battery-runtime estimates for outage operation
- Interactive controls for blower speed, filter loading, filter grade, seal gap, room conditions, and battery settings

## Run locally

Python 3.10+ is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/VAYUVYA-Hub-Filtration-Simulator.git
cd VAYUVYA-Hub-Filtration-Simulator
python -m venv .venv
```

Activate the virtual environment:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies and launch Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run vayuvya_hub_sim.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Model scope and limitations

This is a **design-estimate prototype**, not a validated engineering prediction. The script's docstrings and Model notes tab describe assumptions that should be replaced with measured prototype data. In particular:

- The 2-D planar flow slice is an approximation of the actual hub geometry.
- Blower curve, efficiency, media resistance coefficients, and filter-loading behavior are illustrative placeholders.
- The particle-efficiency model is simplified; activated-carbon gas adsorption is not modelled.
- Leakage estimates should be calibrated using physical leakage or particle-counter tests.
- Room PM calculations assume a simplified, well-mixed room.

Use experimental measurements and engineering review before using outputs for design decisions.

## Main file

- `vayuvya_hub_sim.py` — Streamlit app and simulation calculations

## License

MIT License
