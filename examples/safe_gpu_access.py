#!/usr/bin/env python3
"""
安全的 GPU 数据访问示例

这个版本使用更安全的方法来访问和修改 GPU 数据，避免直接内存操作的错误。
"""

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. GPU data access will be limited.")

from qulacs import QuantumCircuit
from qulacs.gate import H, CNOT, RY

# 尝试导入 GPU 支持
try:
    from qulacs import QuantumStateGpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: QuantumStateGpu not available.")


class SafeGPUAccessor:
    """
    安全的 GPU 数据访问器
    
    使用 Qulacs 提供的接口来安全地访问和修改 GPU 数据
    """
    
    def __init__(self, quantum_state_gpu):
        if not GPU_AVAILABLE:
            raise RuntimeError("QuantumStateGpu is not available")
        
        self.qstate = quantum_state_gpu
        self.qubit_count = quantum_state_gpu.get_qubit_count()
        self.dim = 2 ** self.qubit_count
        
        # 检查是否支持直接 GPU 访问
        self.has_direct_access = hasattr(quantum_state_gpu, 'get_gpu_ptr')
        
    def get_gpu_info(self):
        """获取 GPU 信息"""
        info = {
            'qubit_count': self.qubit_count,
            'dimension': self.dim,
            'device_name': self.qstate.get_device_name()
        }
        
        if self.has_direct_access:
            info.update({
                'gpu_ptr': f"0x{self.qstate.get_gpu_ptr():x}",
                'device_number': self.qstate.get_device_number()
            })
        
        return info
    
    def read_gpu_data(self):
        """
        安全地读取 GPU 数据
        
        Returns:
            numpy.ndarray: CPU 内存中的量子状态数据
        """
        return self.qstate.get_vector()
    
    def write_gpu_data(self, new_data):
        """
        安全地写入 GPU 数据
        
        Args:
            new_data: 新的量子状态数据（numpy 数组或列表）
        """
        if isinstance(new_data, list):
            new_data = np.array(new_data, dtype=np.complex128)
        elif not isinstance(new_data, np.ndarray):
            raise ValueError("Data must be numpy array or list")
        
        if new_data.shape != (self.dim,):
            raise ValueError(f"Data shape {new_data.shape} doesn't match expected {(self.dim,)}")
        
        # 使用 Qulacs 的 load 方法安全地更新 GPU 数据
        self.qstate.load(new_data.tolist())
    
    def apply_operation_on_gpu(self, operation_func, use_cupy=True):
        """
        在 GPU 数据上应用操作
        
        Args:
            operation_func: 操作函数，接受数组并返回修改后的数组
            use_cupy: 是否尝试使用 CuPy 进行 GPU 操作
        """
        if use_cupy and CUPY_AVAILABLE and self.has_direct_access:
            try:
                return self._apply_cupy_operation(operation_func)
            except Exception as e:
                print(f"CuPy 操作失败，回退到 CPU 操作: {e}")
        
        # 回退到 CPU 操作
        return self._apply_cpu_operation(operation_func)
    
    def _apply_cupy_operation(self, operation_func):
        """使用 CuPy 进行 GPU 操作"""
        # 读取当前数据到 CuPy
        cpu_data = self.read_gpu_data()
        cp.cuda.Device(self.qstate.get_device_number()).use()
        gpu_data = cp.asarray(cpu_data)
        
        # 应用操作
        result = operation_func(gpu_data)
        
        # 确保结果在 CPU 上并写回
        if isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)
        
        self.write_gpu_data(result)
        return result
    
    def _apply_cpu_operation(self, operation_func):
        """使用 CPU 进行操作"""
        # 读取数据
        cpu_data = self.read_gpu_data()
        
        # 应用操作
        result = operation_func(cpu_data)
        
        # 写回 GPU
        self.write_gpu_data(result)
        return result
    
    def compute_statistics(self):
        """计算量子状态统计信息"""
        if CUPY_AVAILABLE and self.has_direct_access:
            try:
                return self._compute_gpu_statistics()
            except Exception as e:
                print(f"GPU 统计计算失败，使用 CPU: {e}")
        
        return self._compute_cpu_statistics()
    
    def _compute_gpu_statistics(self):
        """在 GPU 上计算统计信息"""
        cpu_data = self.read_gpu_data()
        cp.cuda.Device(self.qstate.get_device_number()).use()
        gpu_data = cp.asarray(cpu_data)
        
        stats = {
            'norm_squared': float(cp.sum(cp.abs(gpu_data) ** 2)),
            'max_amplitude': float(cp.max(cp.abs(gpu_data))),
            'mean_amplitude': float(cp.mean(cp.abs(gpu_data)))
        }
        
        # 计算熵
        probs = cp.abs(gpu_data) ** 2
        probs = probs[probs > 1e-12]
        if len(probs) > 0:
            entropy = -cp.sum(probs * cp.log2(probs))
            stats['entropy'] = float(entropy)
        else:
            stats['entropy'] = 0.0
        
        return stats
    
    def _compute_cpu_statistics(self):
        """在 CPU 上计算统计信息"""
        data = self.read_gpu_data()
        
        stats = {
            'norm_squared': float(np.sum(np.abs(data) ** 2)),
            'max_amplitude': float(np.max(np.abs(data))),
            'mean_amplitude': float(np.mean(np.abs(data)))
        }
        
        # 计算熵
        probs = np.abs(data) ** 2
        probs = probs[probs > 1e-12]
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log2(probs))
            stats['entropy'] = float(entropy)
        else:
            stats['entropy'] = 0.0
        
        return stats


def example_safe_gpu_access():
    """演示安全的 GPU 数据访问"""
    
    if not GPU_AVAILABLE:
        print("跳过示例：QuantumStateGpu 不可用")
        return
    
    print("=== 安全的 GPU 数据访问示例 ===\n")
    
    # 创建量子电路
    n_qubits = 4
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_RY_gate(1, np.pi/4)
    circuit.add_CNOT_gate(1, 2)
    
    # 创建 GPU 状态
    gpu_state = QuantumStateGpu(n_qubits)
    circuit.update_quantum_state(gpu_state)
    
    # 创建安全访问器
    accessor = SafeGPUAccessor(gpu_state)
    
    # 显示 GPU 信息
    info = accessor.get_gpu_info()
    print("GPU 状态信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 计算统计信息
    print("\n--- 计算统计信息 ---")
    stats = accessor.compute_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
    
    # 应用自定义操作
    print("\n--- 应用自定义操作 ---")
    
    def global_phase_shift(data):
        """应用全局相位"""
        if hasattr(data, 'exp'):  # CuPy 数组
            return data * data.exp(1j * np.pi / 4)
        else:  # NumPy 数组
            return data * np.exp(1j * np.pi / 4)
    
    # 应用操作
    print("应用全局相位变换...")
    accessor.apply_operation_on_gpu(global_phase_shift)
    
    # 验证结果
    new_stats = accessor.compute_statistics()
    print("变换后的统计信息:")
    for key, value in new_stats.items():
        print(f"{key}: {value:.6f}")
    
    # 测试数据读写
    print("\n--- 测试数据读写 ---")
    original_data = accessor.read_gpu_data()
    print(f"读取数据形状: {original_data.shape}")
    print(f"读取数据范数: {np.sum(np.abs(original_data)**2):.6f}")
    
    # 创建新的数据并写入
    new_data = original_data * 0.8  # 缩放
    new_data = new_data / np.linalg.norm(new_data)  # 归一化
    
    accessor.write_gpu_data(new_data)
    print("写入新数据完成")
    
    # 验证写入
    read_back = accessor.read_gpu_data()
    print(f"验证范数: {np.sum(np.abs(read_back)**2):.6f}")
    print(f"数据相等性检查: {np.allclose(new_data, read_back)}")


def example_interoperability_safe():
    """演示与其他库的安全互操作"""
    
    if not (GPU_AVAILABLE and CUPY_AVAILABLE):
        print("跳过互操作示例：需要 GPU 支持和 CuPy")
        return
    
    print("\n=== 安全的互操作性示例 ===\n")
    
    # 创建量子状态
    n_qubits = 3
    gpu_state = QuantumStateGpu(n_qubits)
    gpu_state.set_Haar_random_state()
    
    accessor = SafeGPUAccessor(gpu_state)
    
    # 定义混合操作
    def mix_with_random_state(data):
        """与随机状态混合"""
        if hasattr(data, 'random'):  # CuPy
            random_state = data.random.random(data.shape) + 1j * data.random.random(data.shape)
            combined = 0.6 * data + 0.8 * random_state
            return combined / data.linalg.norm(combined)
        else:  # NumPy
            random_state = np.random.random(data.shape) + 1j * np.random.random(data.shape)
            combined = 0.6 * data + 0.8 * random_state
            return combined / np.linalg.norm(combined)
    
    print("应用随机状态混合...")
    original_norm = accessor.compute_statistics()['norm_squared']
    print(f"原始状态范数: {original_norm:.6f}")
    
    accessor.apply_operation_on_gpu(mix_with_random_state)
    
    new_norm = accessor.compute_statistics()['norm_squared']
    print(f"混合后状态范数: {new_norm:.6f}")
    
    print("混合操作完成")


def performance_comparison():
    """性能比较示例"""
    
    if not GPU_AVAILABLE:
        print("跳过性能比较：GPU 不可用")
        return
    
    print("\n=== 性能比较 ===\n")
    
    import time
    
    # 创建较大的量子状态进行测试
    n_qubits = 6  # 64 维状态
    gpu_state = QuantumStateGpu(n_qubits)
    gpu_state.set_Haar_random_state()
    
    accessor = SafeGPUAccessor(gpu_state)
    
    # CPU 操作测试
    start_time = time.time()
    cpu_stats = accessor._compute_cpu_statistics()
    cpu_time = time.time() - start_time
    
    print(f"CPU 计算时间: {cpu_time:.6f} 秒")
    print(f"CPU 计算范数: {cpu_stats['norm_squared']:.6f}")
    
    if CUPY_AVAILABLE and accessor.has_direct_access:
        # GPU 操作测试
        start_time = time.time()
        try:
            gpu_stats = accessor._compute_gpu_statistics()
            gpu_time = time.time() - start_time
            
            print(f"GPU 计算时间: {gpu_time:.6f} 秒")
            print(f"GPU 计算范数: {gpu_stats['norm_squared']:.6f}")
            
            if gpu_time > 0:
                print(f"加速比: {cpu_time/gpu_time:.2f}x")
            
        except Exception as e:
            print(f"GPU 计算失败: {e}")
    else:
        print("GPU 计算不可用")


if __name__ == "__main__":
    try:
        example_safe_gpu_access()
        example_interoperability_safe()
        performance_comparison()
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n确保你已经:")
        print("1. 重新编译并安装了修改后的 qulacs")
        print("2. 有可用的 CUDA GPU")
        print("3. 可选：安装了 CuPy")
