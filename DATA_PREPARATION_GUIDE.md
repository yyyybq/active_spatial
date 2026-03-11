# Active Spatial 训练数据准备完整指南

## ⚠️ 重要说明

根据代码分析，训练**必须**包含以下字段，缺一不可！

---

## 📋 必需的数据格式（JSONL）

### 完整示例

```json
{
  "scene_id": "scene_001",
  "object_label": "chair",
  "preset": "front",
  "distance": 2.0,
  "init_camera": {
    "intrinsics": [
      [525.0, 0.0, 256.0],
      [0.0, 525.0, 256.0],
      [0.0, 0.0, 1.0]
    ],
    "extrinsics": [
      [1.0, 0.0, 0.0, 0.5],
      [0.0, 1.0, 0.0, -1.0],
      [0.0, 0.0, 1.0, 1.5],
      [0.0, 0.0, 0.0, 1.0]
    ]
  },
  "target_position": [2.0, 0.0, 1.5],
  "camera_params": {
    "forward": [1.0, 0.0, 0.0]
  }
}
```

---

## 🔍 字段详细说明

### 1️⃣ `scene_id` (必需)
- **类型**: string
- **说明**: 场景标识符
- **用途**: 
  - 对应 PLY 文件名（如 `scene_001.ply`）
  - 渲染器用它加载对应的 3D 场景
- **示例**: `"scene_001"`, `"room_living_01"`

### 2️⃣ `object_label` (必需)
- **类型**: string
- **说明**: 目标物体的类别/名称
- **用途**: 构建任务提示 "Move to the front view of the **chair**"
- **示例**: `"chair"`, `"sofa"`, `"table"`, `"tv"`

### 3️⃣ `preset` (必需)
- **类型**: string
- **说明**: 目标视角方向
- **可选值**: `"front"`, `"back"`, `"left"`, `"right"`, `"top"`, `"bottom"`
- **用途**: 构建任务提示 "Move to the **front** view of the chair"
- **示例**: `"front"`

### 4️⃣ `distance` (可选但推荐)
- **类型**: float
- **说明**: 期望的相机到目标物体的距离（米）
- **用途**: 构建任务提示 "about **2.0** meters away"
- **示例**: `2.0`, `1.5`, `3.0`

### 5️⃣ `init_camera` (必需)
包含两个子字段：

#### 5.1 `intrinsics` (必需)
- **类型**: 3x3 矩阵
- **说明**: 相机内参矩阵
- **格式**:
  ```
  [[fx,  0, cx],
   [ 0, fy, cy],
   [ 0,  0,  1]]
  ```
- **参数说明**:
  - `fx, fy`: 焦距（像素单位）
  - `cx, cy`: 主点坐标（通常是图像中心）

**计算方法**:
```python
import numpy as np

def generate_intrinsics(width=512, height=512, fov_deg=60):
    """生成相机内参"""
    fov_rad = np.radians(fov_deg)
    fx = fy = width / (2 * np.tan(fov_rad / 2))
    cx = width / 2.0
    cy = height / 2.0
    return [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]
```

#### 5.2 `extrinsics` (必需)
- **类型**: 4x4 矩阵
- **说明**: 相机外参矩阵（Camera-to-World 变换）
- **格式**:
  ```
  [[R11, R12, R13, tx],
   [R21, R22, R23, ty],
   [R31, R32, R33, tz],
   [  0,   0,   0,  1]]
  ```
- **参数说明**:
  - `R`: 3x3 旋转矩阵（相机在世界坐标系中的朝向）
  - `t`: 3D 平移向量 [tx, ty, tz]（相机在世界坐标系中的位置）

**坐标系约定（OpenGL）**:
- +X: 右
- +Y: 上  
- -Z: 前（相机朝向）

**计算方法**:
```python
def look_at_matrix(camera_position, look_at_point, up_vector=[0, 0, 1]):
    """
    生成相机外参矩阵（c2w）
    
    Args:
        camera_position: 相机位置 [x, y, z]
        look_at_point: 相机看向的点 [x, y, z]
        up_vector: 上方向 [x, y, z]
    
    Returns:
        4x4 c2w 矩阵
    """
    position = np.array(camera_position, dtype=np.float64)
    target = np.array(look_at_point, dtype=np.float64)
    up = np.array(up_vector, dtype=np.float64)
    
    # 计算相机坐标系的三个轴
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up_new = np.cross(right, forward)
    up_new = up_new / np.linalg.norm(up_new)
    
    # OpenGL 约定: -Z 是前方
    R = np.column_stack([right, up_new, -forward])
    
    # 构建 4x4 变换矩阵
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = position
    
    return c2w.tolist()
```

### 6️⃣ `target_position` (必需)
- **类型**: [x, y, z] 数组
- **说明**: 目标相机应该到达的位置（米）
- **用途**: 
  - 计算距离奖励
  - 判断是否成功
- **示例**: `[2.0, 0.0, 1.5]`

### 7️⃣ `camera_params.forward` (必需)
- **类型**: [dx, dy, dz] 数组
- **说明**: 目标相机应该朝向的方向（单位向量）
- **用途**: 
  - 计算朝向奖励
  - 评估相机是否正确对准物体
- **示例**: `[1.0, 0.0, 0.0]` (朝向 +X 方向)

**常见方向**:
- 前: `[1, 0, 0]` 或 `[0, 1, 0]` (取决于场景坐标系)
- 后: `[-1, 0, 0]` 或 `[0, -1, 0]`
- 左: `[0, 1, 0]` 或 `[-1, 0, 0]`
- 右: `[0, -1, 0]` 或 `[1, 0, 0]`
- 下: `[0, 0, -1]`
- 上: `[0, 0, 1]`

---

## 🛠️ 完整数据生成脚本

```python
#!/usr/bin/env python3
"""
生成 Active Spatial 训练数据
确保包含所有必需字段！
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


def generate_intrinsics(width=512, height=512, fov_deg=60):
    """生成相机内参"""
    fov_rad = np.radians(fov_deg)
    fx = fy = width / (2 * np.tan(fov_rad / 2))
    cx = width / 2.0
    cy = height / 2.0
    return [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0]
    ]


def look_at_matrix(camera_position, look_at_point, up_vector=[0, 0, 1]):
    """
    生成相机外参矩阵（c2w）
    
    Args:
        camera_position: 相机位置 [x, y, z]
        look_at_point: 相机看向的点 [x, y, z]
        up_vector: 上方向 [x, y, z]
    
    Returns:
        4x4 c2w 矩阵（列表格式）
    """
    position = np.array(camera_position, dtype=np.float64)
    target = np.array(look_at_point, dtype=np.float64)
    up = np.array(up_vector, dtype=np.float64)
    
    # 计算相机坐标系的三个轴
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up_new = np.cross(right, forward)
    up_new = up_new / np.linalg.norm(up_new)
    
    # OpenGL 约定: -Z 是前方
    R = np.column_stack([right, up_new, -forward])
    
    # 构建 4x4 变换矩阵
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = position
    
    return c2w.tolist()


def get_preset_config(preset: str, object_center: List[float], distance: float) -> Tuple[List[float], List[float]]:
    """
    根据 preset 计算目标位置和朝向
    
    Args:
        preset: 视角类型 (front/back/left/right/top/bottom)
        object_center: 物体中心位置 [x, y, z]
        distance: 期望距离
    
    Returns:
        (target_position, forward_direction)
    """
    obj_x, obj_y, obj_z = object_center
    
    preset_configs = {
        "front": {
            "offset": [distance, 0, 0],
            "forward": [-1.0, 0.0, 0.0]
        },
        "back": {
            "offset": [-distance, 0, 0],
            "forward": [1.0, 0.0, 0.0]
        },
        "left": {
            "offset": [0, distance, 0],
            "forward": [0.0, -1.0, 0.0]
        },
        "right": {
            "offset": [0, -distance, 0],
            "forward": [0.0, 1.0, 0.0]
        },
        "top": {
            "offset": [0, 0, distance],
            "forward": [0.0, 0.0, -1.0]
        },
        "bottom": {
            "offset": [0, 0, -distance],
            "forward": [0.0, 0.0, 1.0]
        }
    }
    
    if preset not in preset_configs:
        raise ValueError(f"Unknown preset: {preset}")
    
    config = preset_configs[preset]
    offset = config["offset"]
    forward = config["forward"]
    
    target_position = [
        obj_x + offset[0],
        obj_y + offset[1],
        obj_z + offset[2]
    ]
    
    return target_position, forward


def generate_training_sample(
    scene_id: str,
    object_label: str,
    preset: str,
    object_center: List[float],
    distance: float = 2.0,
    init_position: List[float] = None,
    image_width: int = 512,
    image_height: int = 512
) -> dict:
    """
    生成一个训练样本
    
    Args:
        scene_id: 场景 ID
        object_label: 物体标签
        preset: 视角预设
        object_center: 物体中心位置
        distance: 目标距离
        init_position: 初始相机位置（如为 None 则随机生成）
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        完整的训练样本字典
    """
    # 计算目标位置和朝向
    target_position, target_forward = get_preset_config(preset, object_center, distance)
    
    # 生成初始相机位置（随机或指定）
    if init_position is None:
        # 在物体周围随机生成
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(1.0, 4.0)
        init_position = [
            object_center[0] + radius * np.cos(angle),
            object_center[1] + radius * np.sin(angle),
            object_center[2] + np.random.uniform(-0.5, 0.5)
        ]
    
    # 生成相机内参
    intrinsics = generate_intrinsics(image_width, image_height)
    
    # 生成相机外参（初始相机看向物体中心）
    extrinsics = look_at_matrix(init_position, object_center)
    
    # 构建完整样本
    sample = {
        "scene_id": scene_id,
        "object_label": object_label,
        "preset": preset,
        "distance": float(distance),
        "init_camera": {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics
        },
        "target_position": [float(x) for x in target_position],
        "camera_params": {
            "forward": [float(x) for x in target_forward]
        }
    }
    
    return sample


def generate_dataset(
    scene_configs: List[dict],
    output_path: str = "train_data.jsonl"
):
    """
    批量生成训练数据集
    
    Args:
        scene_configs: 场景配置列表
        output_path: 输出文件路径
    
    scene_configs 格式示例:
    [
        {
            "scene_id": "scene_001",
            "objects": [
                {"label": "chair", "center": [0, 0, 0.5]},
                {"label": "table", "center": [2, 0, 0.8]}
            ]
        },
        ...
    ]
    """
    samples = []
    presets = ["front", "back", "left", "right"]
    distances = [1.5, 2.0, 2.5]
    
    for scene_config in scene_configs:
        scene_id = scene_config["scene_id"]
        objects = scene_config["objects"]
        
        for obj in objects:
            object_label = obj["label"]
            object_center = obj["center"]
            
            for preset in presets:
                for distance in distances:
                    # 为每个配置生成多个样本（不同初始位置）
                    for _ in range(2):  # 每个配置生成2个样本
                        sample = generate_training_sample(
                            scene_id=scene_id,
                            object_label=object_label,
                            preset=preset,
                            object_center=object_center,
                            distance=distance
                        )
                        samples.append(sample)
    
    # 保存为 JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ 生成 {len(samples)} 个训练样本")
    print(f"✅ 保存到: {output_path}")
    
    # 打印第一个样本作为参考
    print(f"\n示例样本:")
    print(json.dumps(samples[0], indent=2))
    
    return samples


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 定义场景配置
    scene_configs = [
        {
            "scene_id": "scene_001",
            "objects": [
                {"label": "chair", "center": [0, 0, 0.5]},
                {"label": "sofa", "center": [3, 0, 0.6]},
                {"label": "table", "center": [1.5, 1.5, 0.8]}
            ]
        },
        {
            "scene_id": "scene_002",
            "objects": [
                {"label": "bed", "center": [0, 0, 0.4]},
                {"label": "desk", "center": [2, 1, 0.75]},
                {"label": "chair", "center": [2, 0, 0.5]}
            ]
        },
        {
            "scene_id": "scene_003",
            "objects": [
                {"label": "tv", "center": [0, 0, 1.2]},
                {"label": "sofa", "center": [3, 0, 0.6]}
            ]
        }
    ]
    
    # 生成训练集
    generate_dataset(
        scene_configs=scene_configs,
        output_path="data/active_spatial/train_data.jsonl"
    )
    
    # 生成测试集（使用不同的随机种子）
    np.random.seed(999)
    generate_dataset(
        scene_configs=scene_configs[:1],  # 只用一个场景做测试
        output_path="data/active_spatial/test_data.jsonl"
    )
```

---

## ✅ 验证数据格式

运行此脚本验证生成的数据：

```python
#!/usr/bin/env python3
"""验证 JSONL 数据格式"""

import json
from pathlib import Path

def validate_sample(sample: dict, line_num: int) -> bool:
    """验证单个样本"""
    required_fields = {
        "scene_id": str,
        "object_label": str,
        "preset": str,
        "distance": (int, float),
        "init_camera": dict,
        "target_position": list,
        "camera_params": dict
    }
    
    errors = []
    
    # 检查必需字段
    for field, expected_type in required_fields.items():
        if field not in sample:
            errors.append(f"缺少字段: {field}")
        elif not isinstance(sample[field], expected_type):
            errors.append(f"字段 {field} 类型错误: 期望 {expected_type}, 实际 {type(sample[field])}")
    
    # 检查嵌套字段
    if "init_camera" in sample:
        if "intrinsics" not in sample["init_camera"]:
            errors.append("缺少 init_camera.intrinsics")
        if "extrinsics" not in sample["init_camera"]:
            errors.append("缺少 init_camera.extrinsics")
    
    if "camera_params" in sample:
        if "forward" not in sample["camera_params"]:
            errors.append("缺少 camera_params.forward")
    
    # 检查数组长度
    if "target_position" in sample and len(sample["target_position"]) != 3:
        errors.append(f"target_position 应该是 3 元素数组")
    
    if errors:
        print(f"❌ 第 {line_num} 行错误:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    return True


def validate_jsonl_file(file_path: str):
    """验证整个 JSONL 文件"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    print(f"验证文件: {file_path}")
    print("=" * 60)
    
    valid_count = 0
    error_count = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                if validate_sample(sample, line_num):
                    valid_count += 1
                else:
                    error_count += 1
            except json.JSONDecodeError as e:
                print(f"❌ 第 {line_num} 行 JSON 解析错误: {e}")
                error_count += 1
    
    print("=" * 60)
    print(f"总计: {valid_count + error_count} 行")
    print(f"✅ 有效: {valid_count}")
    print(f"❌ 错误: {error_count}")
    
    return error_count == 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python validate_data.py <jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid = validate_jsonl_file(file_path)
    
    sys.exit(0 if is_valid else 1)
```

---

## 🎯 快速开始

```bash
# 1. 生成数据
python generate_training_data.py

# 2. 验证数据
python validate_data.py data/active_spatial/train_data.jsonl

# 3. 检查输出
head -1 data/active_spatial/train_data.jsonl | python -m json.tool

# 4. 更新配置
vim scripts/examples/vagen_base/active_spatial/env_config.yaml
# 设置 jsonl_path: "data/active_spatial/train_data.jsonl"
# 设置 gs_root: "/path/to/your/gaussian_splatting_files"

# 5. 开始训练
bash scripts/examples/vagen_base/active_spatial/run.sh
```

---

## 📌 总结

**所有必需字段（缺一不可）**:
1. ✅ `scene_id` - 场景 ID
2. ✅ `object_label` - 物体标签
3. ✅ `preset` - 视角预设
4. ✅ `distance` - 目标距离
5. ✅ `init_camera.intrinsics` - 相机内参 (3x3)
6. ✅ `init_camera.extrinsics` - 相机外参 (4x4)
7. ✅ `target_position` - 目标位置 [x, y, z]
8. ✅ `camera_params.forward` - 目标朝向 [dx, dy, dz]

**不要遗漏任何字段！** 代码会用到每一个字段来计算奖励和判断成功。
