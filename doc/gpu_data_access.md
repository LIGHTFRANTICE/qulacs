# 在其他 Python 程序中直接访问 Qulacs GPU 数据

本文档介绍如何在其他 Python 程序中直接访问存储在 GPU 中的 Qulacs 量子状态数据，而无需将数据复制到 CPU 内存。

## 前提条件

1. 安装支持 GPU 的 Qulacs 版本
2. 安装 CUDA 驱动和运行时
3. 可选：安装 CuPy 或 PyCUDA 用于更高级的 GPU 操作

## 方法概述

我们提供了几种方法来直接访问 GPU 数据：

### 方法 1: 使用修改后的 Qulacs 接口

我们在 `QuantumStateGpu` 类中添加了新的方法：

- `get_gpu_ptr()`: 返回 GPU 数据指针（作为整数）
- `get_device_number()`: 返回 GPU 设备号

#### 修改的源代码

**1. 在 `src/cppsim/state_gpu.hpp` 中添加：**

```cpp
/**
 * \~japanese-en GPU上の量子状态データの生ポインタを取得する
 *
 * @return GPU上の複素ベクトルのポインタ
 */
virtual void* data_gpu() const {
    return this->_state_vector;
}

/**
 * \~japanese-en GPU上の量子状態データのアドレスを整数として取得する
 *
 * @return GPU上のデータアドレス
 */
virtual uintptr_t data_gpu_ptr() const {
    return reinterpret_cast<uintptr_t>(this->_state_vector);
}
```

**2. 在 `python/cppsim_wrapper.cpp` 中添加 Python 绑定：**

```cpp
.def(
    "get_gpu_ptr",
    [](const QuantumStateGpu& state) -> uintptr_t {
        return state.data_gpu_ptr();
    },
    "Get GPU data pointer as integer")
.def(
    "get_device_number",
    [](const QuantumStateGpu& state) -> UINT {
        return state.device_number;
    },
    "Get GPU device number")
```

### 方法 2: 使用 CuPy 直接访问

如果你安装了 CuPy，可以使用 `examples/gpu_data_access.py` 中的 `GPUQuantumStateAccessor` 类：

```python
from qulacs import QuantumStateGpu
import cupy as cp

# 创建量子状态
gpu_state = QuantumStateGpu(n_qubits)

# 创建访问器
accessor = GPUQuantumStateAccessor(gpu_state)

# 获取 CuPy 数组
gpu_array = accessor.get_cupy_array()

# 直接在 GPU 上操作
result = cp.sum(cp.abs(gpu_array) ** 2)  # 计算范数
```

### 方法 3: 使用 ctypes 和 CUDA Runtime API

使用 `examples/ctypes_gpu_access.py` 中的 `GPUDataInterface` 类：

```python
from qulacs import QuantumStateGpu

# 创建量子状态
gpu_state = QuantumStateGpu(n_qubits)

# 创建接口
interface = GPUDataInterface(gpu_state)

# 复制到主机进行操作
host_data = interface.copy_to_host()

# 修改数据
modified_data = host_data * np.exp(1j * np.pi/4)

# 复制回 GPU
interface.copy_from_host(modified_data)
```

## 编译和安装

### 1. 重新编译 Qulacs

```bash
# 确保 GPU 支持已启用
export USE_GPU=Yes

# 编译
./script/build_gcc_with_gpu.sh

# 安装
pip install .
```

### 2. 安装依赖

```bash
# 安装 CuPy（推荐）
pip install cupy

# 或者安装 PyCUDA
pip install pycuda
```

## 使用示例

### 基本使用

```python
from qulacs import QuantumStateGpu, QuantumCircuit
from qulacs.gate import H, CNOT

# 创建量子电路
circuit = QuantumCircuit(3)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)

# 创建 GPU 状态
gpu_state = QuantumStateGpu(3)
circuit.update_quantum_state(gpu_state)

# 获取 GPU 指针信息
if hasattr(gpu_state, 'get_gpu_ptr'):
    ptr = gpu_state.get_gpu_ptr()
    device = gpu_state.get_device_number()
    print(f"GPU 指针: 0x{ptr:x}, 设备: {device}")
else:
    print("请更新 Qulacs 以支持直接 GPU 访问")
```

### 与其他 GPU 库集成

```python
import cupy as cp
from gpu_data_access import GPUQuantumStateAccessor

# 创建访问器
accessor = GPUQuantumStateAccessor(gpu_state)
qulacs_array = accessor.get_cupy_array()

# 与其他 CuPy 数组交互
other_array = cp.random.random(qulacs_array.shape, dtype=cp.complex128)

# 计算内积
inner_prod = cp.vdot(qulacs_array, other_array)

# 线性组合
combined = 0.6 * qulacs_array + 0.8 * other_array
combined /= cp.linalg.norm(combined)

# 更新原始状态
qulacs_array[:] = combined
```

## 性能考虑

1. **避免频繁的 CPU-GPU 传输**：尽量在 GPU 上完成所有计算
2. **内存对齐**：确保数据结构正确对齐以获得最佳性能
3. **异步操作**：使用 CUDA 流进行异步操作
4. **内存管理**：小心管理 GPU 内存以避免泄漏

## 注意事项

1. **内存安全**：直接访问 GPU 内存需要小心，确保不会访问无效内存
2. **设备同步**：在访问 GPU 数据前确保所有操作已完成
3. **兼容性**：这些功能需要特定版本的 Qulacs 和 CUDA
4. **调试困难**：GPU 内存访问错误可能导致程序崩溃

## 故障排除

### 常见问题

1. **"QuantumStateGpu 不支持 get_gpu_ptr"**
   - 重新编译 Qulacs 确保包含了新的修改

2. **"CUDA Runtime 不可用"**
   - 检查 CUDA 安装
   - 确保 libcudart.so 在系统路径中

3. **"CuPy 导入失败"**
   - 安装与你的 CUDA 版本兼容的 CuPy

### 验证安装

```python
def verify_gpu_access():
    try:
        from qulacs import QuantumStateGpu
        state = QuantumStateGpu(2)
        
        if hasattr(state, 'get_gpu_ptr'):
            print("✓ GPU 直接访问可用")
            print(f"  GPU 指针: 0x{state.get_gpu_ptr():x}")
            print(f"  设备号: {state.get_device_number()}")
        else:
            print("✗ GPU 直接访问不可用")
            
    except Exception as e:
        print(f"✗ 错误: {e}")

verify_gpu_access()
```

## 进一步阅读

- [Qulacs 官方文档](http://docs.qulacs.org)
- [CuPy 文档](https://cupy.dev/)
- [CUDA Python 文档](https://nvidia.github.io/cuda-python/)
- [PyCUDA 文档](https://documen.tician.de/pycuda/)
