## 📦 项目结构概览

```
│   camera_test.py                 # 渲染测试脚本（单独测试Gaussian渲染）
│   ply_gaussian_loader.py         # PLY格式Gaussian点云加载器
│   camera_position_data.jsonl     # 相机位置与目标参数数据
│   requirements.txt / setup.py    # 环境依赖与安装脚本
│
├── tool_test                      # gym_approach渲染输出目录
│       step.png
│
├── view_suite/
│   ├── envs/                      # 环境定义模块
│   │   └── active_spatial_intelligence/
│   │       ├── gym_approach.py    # 主环境：交互逻辑、动作解析、reward计算
│   │       └── data_gen/
│   │           ├── camera_generation.py
│   │           └── tv_approach_video.py
│   │
│   ├── interiorGS/                # Gaussian渲染模块
│   │   ├── gym_gs_render.py
│   │   ├── unified_renderer.py
│   │   ├── view_manipulator.py    # 相机控制类（平移、旋转、重置）
│   │   ├── render/
│   │   │       base_render.py
│   │   │       gs_render.py
│   │   ├── service/               # 渲染服务端（WebSocket）
│   │   │       gs_render_service.py
│   │   │       gs_render_client.py
│   │   │       gs_render_handler.py
│   │   └── test/
│   │           test_unified_render.py
│   │
│   ├── evaluate/（TODO）           # 评估逻辑与适配器
│   │   ├── adapters/
│   │   ├── conf/
│   │   └── utils/
│   │
│   ├── gym/（TODO）               # 通用Gym封装与代理模块
│   │   ├── gym_base_env.py
│   │   ├── gym_image_env.py
│   │   └── service/
│   │
│   └── service/                   # 通用WebSocket服务
│       ├── ws_client.py
│       ├── ws_endpoint_factory.py
│       ├── handler/
│       └── utils/
│
└── __pycache__                    # Python缓存文件
```

---

## ⚙️ 安装说明

### 依赖环境

- Python ≥ 3.11  
- PyTorch ≥ 2.1  
- OpenGL / EGL（支持无头渲染）  
- CUDA（推荐，用于GPU加速）  

### 安装命令

```bash
git clone https://github.com/xxx/ViewSuite.git
cd ViewSuite
pip install -e .
```

主要依赖包括：

```
numpy, torch, open3d, fastapi, uvicorn, websockets, imageio
```

---

## 🚀 快速上手

### 🖥️ 启动远程渲染服务（Server）

在服务器上运行：

```bash
python -m view_suite.interiorGS.service.gs_render_service     --host 0.0.0.0     --port 8766     --interiorgs_root ../dataset
```

说明：

- 默认监听端口 `8766`
- `dataset/` 中应包含 `.ply` 文件（如 `3dgs_compressed.ply`）
- 渲染服务基于 FastAPI + WebSocket，提供远程渲染能力

---

### 💻 启动交互环境（Client）

在本地或另一台服务器运行：

只测试交互：
```bash
python -m view_suite.envs.active_spatial_intelligence.gym_approach     --jsonl_path "camera_position_data.jsonl"     --render_backend client     --client_url "ws://115.182.62.248:8766/render/interiorgs"     --client_origin "http://localhost"     --scene_id "3dgs_compressed"

```
或者运行approach任务，包含reward和prompt的：
 python -m view_suite.envs.active_spatial_intelligence.gym     --jsonl_path "datatest.jsonl"     --render_backend client     --client_url "ws://127.0.0.1:8766/render/interiorgs"     --client_origin "http://localhost"     --scene_id "3dgs_compressed"
 
参数说明：

| 参数               | 含义                                              |
| ------------------ | ------------------------------------------------- |
| `--jsonl_path`     | 相机数据路径，格式如 `camera_position_data.jsonl` |
| `--render_backend` | 设置为 `client` 以连接远程渲染服务                |
| `--client_url`     | 渲染服务器的地址与端口                            |
| `--client_origin`  | WebSocket 客户端来源，用于跨域                    |
| `--scene_id`       | `.ply` 文件名（去除后缀）                         |

系统将：

- 从 JSONL 文件中提取 `object_label`, `preset`, `distance` 生成任务 prompt；
- 计算目标方向 reward；
- 以 `init_camera` 参数初始化相机（可能需手动检查）。

输出图片将保存至 `tool_test/step.png`。

---

## 🧠 模块功能简述

| 模块                     | 功能说明                                         |
| ------------------------ | ------------------------------------------------ |
| `gym_approach.py`        | 交互环境核心：动作解析、状态更新、奖励函数       |
| `view_manipulator.py`    | 相机控制类（移动、旋转、视角重置）               |
| `gs_render.py`           | 基于 Gaussian Splatting 的图像渲染核心逻辑       |
| `gs_render_service.py`   | 渲染服务端，通过 WebSocket 接收并返回渲染结果    |
| `ply_gaussian_loader.py` | PLY 点云加载、解析高斯参数（位置、颜色、尺度等） |
| `camera_generation.py`   | 批量生成相机位姿与目标方向数据                   |
| `tv_approach_video.py`   | 视频生成与渲染展示示例                           |

---

## 📡 Gym-as-a-Service 架构

整体采用三层架构：

1. **Gym 环境层**（`envs/active_spatial_intelligence/`）  
   - 负责动作解析、状态更新、调用渲染引擎。
2. **渲染服务层**（`interiorGS/service/`）  
   - 提供远程 WebSocket 渲染接口。
3. **通信封装层**（`service/`）  
   - 实现统一的 WebSocket 客户端、服务端和协议管理。

---

## 🧩 当前进展与计划

| 模块                               | 状态     | 说明                              |
| ---------------------------------- | -------- | --------------------------------- |
| `envs.active_spatial_intelligence` | ✅ 完成   | 支持远程渲染与奖励机制            |
| `interiorGS.service`               | ✅ 完成   | 提供 FastAPI + WebSocket 渲染服务 |
| `evaluate`                         | 🕓 开发中 | 计划用于模型评估与agent性能分析   |
| `gym`                              | 🕓 开发中 | 将引入标准化agent控制接口         |

---

## 🖼️ 示例输出

成功执行后，渲染结果会自动保存至：tool_test/step.png