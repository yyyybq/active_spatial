# ViewSuite

A toolkit for 3D scene understanding and novel view generation with a WebSocket-based Gym-as-a-Service infrastructure. It supports local use and distributed rendering/interactive gym environments.

## Installation

### Requirements
- Python >= 3.11
- OpenGL/EGL for offscreen rendering
- CUDA (optional, recommended for GPU acceleration)

### Install from source

```bash
git clone https://github.com/JamesKrW/ViewSuite.git
cd ViewSuite
pip install -e .
```

Core dependencies are defined in `setup.py`: numpy, requests, open3d==0.19.0, uvicorn, fastapi, websockets==15.0.1.

## Quickstart

Two minimal runnable examples are provided below: a purely local gym and a networked ScanNet render service.

### Example A: Run the Sokoban text+image gym (local)

```bash
python view_suite/gym_sokoban/gym_sokoban.py
```

The script provides a CLI loop and saves images under `./test/`.

### Example B: Start the ScanNet render service (WebSocket)

1) Prepare the dataset (see Dataset Setup) and set `SCANNET_ROOT` (points to the directory that contains `scans/`, or an equivalent root).

2) Launch the service:

```bash
python -m view_suite.render_service_scannet.render_service run \
  --scannet_root "${SCANNET_ROOT}" \
  --host 0.0.0.0 --port 8766 \
  --num_shards 8 --max_renderers_per_worker 8 \
  --reload False --log_level info
```

This starts a FastAPI + Uvicorn app exposing the WebSocket endpoint at `ws://<host>:<port>/render`.

A reference client is available in `view_suite/common/service/ws_client.py` to connect, manage requests, and follow the fixed-order protocol described below.



## Service Architecture and Protocol (Gym-as-a-Service)

- Fixed-order WebSocket protocol:
  - Client sends: JSON metadata + binary payload (empty bytes if none).
  - Server responds: JSON metadata + binary payload (empty bytes if none).
  - Each request carries a unique `req_id` for correlation.

- Handler system:
  - `BaseHandler` defines the interface; `RouterHandler` routes by the `op` field.
  - Specialized handlers exist for domains like rendering and gym control.

- Key service components:
  - `ws_endpoint_factory.py`: Creates standardized WS endpoints from handlers, manages lifecycle and errors.
  - `ws_client.py`: Client utility for connection management and request/response correlation with tunable parameters.

- Typical three-tier implementation per gym:
  1) Local implementation (`gym_local.py`): single-process core logic and state (development/test).
  2) Service implementation (`gym_service.py`): WebSocket wrapper, session/concurrency management.
  3) Client implementation (`gym_client.py`): remote-equivalent interface with robust networking.

## Run other environments/benchmarks

- Visual spatial ability benchmark (no-tool version):

```bash
python view_suite/gym_view_spatial_bench/gym_view_spatial_no_tool.py \
  --jsonl_path /path/to/your_data.jsonl \
  --save_path ./test
```

The script prints instructions to the console and saves images to `save_path`.

## Project Structure

```
ViewSuite/
├── view_suite/                         # Main package
│   ├── common/                         # Shared components
│   │   ├── service/                    # Gym-as-a-Service infrastructure
│   │   │   ├── handler/                # Request handlers
│   │   │   ├── ws_client.py            # WebSocket client
│   │   │   └── ws_endpoint_factory.py
│   │   └── scannet/                    # ScanNet-specific components
│   │       ├── render/                 # Render engines (mesh/point cloud)
│   │       ├── service/                # Service handlers for render/gym
│   │       └── utils/                  # Data/pose utilities
│   ├── gym_sokoban/                    # Sokoban environment
│   ├── gym_scannet_forward_dynamics/   # ScanNet forward dynamics gym
│   ├── gym_scannet_inverse_dynamics/   # ScanNet inverse dynamics gym
│   ├── gym_view_spatial_bench/         # Visual spatial benchmark
│   └── render_service_scannet/         # ScanNet render service (FastAPI/WS)
├── setup.py                            # Packaging and dependencies
├── download-scannet.py                 # Dataset download helper
└── README.md
```


## License & Acknowledgements

- Datasets (e.g., ScanNet) are subject to their official licenses and terms.
- Third-party libraries follow their respective licenses—thanks to their authors.


