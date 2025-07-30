#!/usr/bin/env python3
"""
简化的 GPU 数据访问示例

这个示例展示了基本的 GPU 数据访问方法
"""

import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import H, CNOT, RY

# 尝试导入 qulacs GPU 支持
try:
    from qulacs import QuantumStateGpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("警告: QuantumStateGpu 不可用。请确保安装了支持 GPU 的 qulacs 版本。")

def basic_gpu_access_example():
    """基本的 GPU 数据访问示例"""
    
    if not GPU_AVAILABLE:
        print("跳过 GPU 示例（GPU 支持不可用）")
        return
    
    print("=== 基本 GPU 数据访问示例 ===\n")
    
    # 创建量子电路
    n_qubits = 3
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_RY_gate(1, np.pi/4)
    
    # 创建 GPU 状态
    gpu_state = QuantumStateGpu(n_qubits)
    circuit.update_quantum_state(gpu_state)
    
    print(f"量子比特数: {n_qubits}")
    print(f"状态维度: {2**n_qubits}")
    
    # 检查是否有新的 GPU 访问方法
    if hasattr(gpu_state, 'get_gpu_ptr'):
        print(f"GPU 数据指针: 0x{gpu_state.get_gpu_ptr():x}")
        print(f"GPU 设备号: {gpu_state.get_device_number()}")
    else:
        print("GPU 直接访问方法不可用，请重新编译 qulacs")
    
    # 传统方法：获取数据到 CPU
    print("\n--- 传统方法（复制到 CPU）---")
    cpu_vector = gpu_state.get_vector()
    print(f"CPU 向量形状: {cpu_vector.shape}")
    print(f"状态范数: {np.sum(np.abs(cpu_vector)**2):.6f}")
    print(f"前5个元素: {cpu_vector[:5]}")

if __name__ == "__main__":
    basic_gpu_access_example()
