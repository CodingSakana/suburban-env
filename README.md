# suburban-env (fork)

This project is a fork of [suburban-env](https://gitee.com/sukiet/suburban-env).

Quickstart
- Create a virtual environment and install dependencies:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run a quick debug round (CPU, no training):
  - `python -m run mode=debug device=cpu size=128`
- Start the Flask API:
  - `python -m flask_run`
  - Examples:
    - `GET /` health check
    - `GET /cmd/random` random step
    - `GET /current_obs` current observation (JPEG)

Conda setup (alternative)
- Create and activate env:
  - `conda create -n suburban-env python=3.10 -y`
  - `conda activate suburban-env`
- Install deps:
  - `pip install -r requirements.txt`
- Optional: install CUDA-enabled PyTorch via conda if you have NVIDIA GPU drivers:
  - `conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia`
- Run the same commands as above:
  - `python -m run mode=debug device=cpu size=128`
  - or training: `python -m run mode=train device=cuda size=84`

Notes
- Training (`mode=train`) uses the local OmniSafe code in this repo and requires the packages in `requirements.txt`. No external OmniSafe install is needed.
- `opencv-python-headless` is used for servers/containers. If you want OpenCV GUI windows, replace with `opencv-python`.
- `evaluate.py` has a hard-coded Windows path (`LOG_DIR`); set it to your own run directory before using.
