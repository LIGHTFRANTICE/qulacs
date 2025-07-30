#!/usr/bin/env python3
"""
使用 ctypes 直接访问 Qulacs GPU 数据

这个模块提供了使用 ctypes 和 CUDA Runtime API 直接访问 GPU 数据的方法
"""

import ctypes
import numpy as np
from typing import Optional, Tuple

# CUDA Runtime API 函数定义
try:
    # 尝试加载 CUDA Runtime 库
    cuda_runtime = ctypes.CDLL('libcudart.so')
    CUDA_AVAILABLE = True
except OSError:
    try:
        # Windows 上的库名
        cuda_runtime = ctypes.CDLL('cudart64_110.dll')
        CUDA_AVAILABLE = True
    except OSError:
        CUDA_AVAILABLE = False
        print("警告: CUDA Runtime 库不可用")

if CUDA_AVAILABLE:
    # 定义 CUDA 函数
    cudaMemcpy = cuda_runtime.cudaMemcpy
    cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudaMemcpy.restype = ctypes.c_int
    
    cudaDeviceSynchronize = cuda_runtime.cudaDeviceSynchronize
    cudaDeviceSynchronize.argtypes = []
    cudaDeviceSynchronize.restype = ctypes.c_int
    
    # CUDA 内存复制方向常量
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3


class CUDAError(Exception):
    """CUDA 操作错误"""
    pass


def check_cuda_error(error_code: int) -> None:
    """检查 CUDA 错误码"""
    if error_code != 0:
        raise CUDAError(f"CUDA error: {error_code}")


class GPUDataInterface:
    """
    GPU 数据接口类
    
    提供直接访问 Qulacs QuantumStateGpu 数据的方法
    """
    
    def __init__(self, quantum_state_gpu):
        """
        初始化 GPU 数据接口
        
        Args:
            quantum_state_gpu: QuantumStateGpu 实例
        """
        if not hasattr(quantum_state_gpu, 'get_gpu_ptr'):
            raise RuntimeError("QuantumStateGpu does not support get_gpu_ptr method")
        
        self.qstate = quantum_state_gpu
        self.gpu_ptr = quantum_state_gpu.get_gpu_ptr()
        self.device_number = quantum_state_gpu.get_device_number()
        self.qubit_count = quantum_state_gpu.get_qubit_count()
        self.dim = 2 ** self.qubit_count
        self.data_size = self.dim * 16  # complex128 = 16 bytes
        
    def copy_to_host(self) -> np.ndarray:
        """
        将 GPU 数据复制到主机内存
        
        Returns:
            numpy.ndarray: 包含量子状态的复数数组
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA Runtime not available")
        
        # 分配主机内存
        host_array = np.zeros(self.dim, dtype=np.complex128)
        host_ptr = host_array.ctypes.data_as(ctypes.c_void_p)
        
        # 从 GPU 复制到主机
        error_code = cudaMemcpy(
            host_ptr,
            ctypes.c_void_p(self.gpu_ptr),
            self.data_size,
            cudaMemcpyDeviceToHost
        )
        check_cuda_error(error_code)
        
        return host_array
    
    def copy_from_host(self, host_array: np.ndarray) -> None:
        """
        将主机数据复制到 GPU
        
        Args:
            host_array: 要复制的主机数组
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA Runtime not available")
        
        if host_array.shape != (self.dim,):
            raise ValueError(f"Array shape {host_array.shape} doesn't match expected {(self.dim,)}")
        
        if host_array.dtype != np.complex128:
            host_array = host_array.astype(np.complex128)
        
        host_ptr = host_array.ctypes.data_as(ctypes.c_void_p)
        
        # 从主机复制到 GPU
        error_code = cudaMemcpy(
            ctypes.c_void_p(self.gpu_ptr),
            host_ptr,
            self.data_size,
            cudaMemcpyHostToDevice
        )
        check_cuda_error(error_code)
    
    def synchronize(self) -> None:
        """同步 GPU 操作"""
        if CUDA_AVAILABLE:
            error_code = cudaDeviceSynchronize()
            check_cuda_error(error_code)
    
    def get_gpu_pointer_info(self) -> dict:
        """
        获取 GPU 指针信息
        
        Returns:
            dict: 包含 GPU 指针信息的字典
        """
        return {
            'gpu_ptr': self.gpu_ptr,
            'gpu_ptr_hex': f"0x{self.gpu_ptr:x}",
            'device_number': self.device_number,
            'qubit_count': self.qubit_count,
            'dimension': self.dim,
            'data_size_bytes': self.data_size
        }


def create_gpu_memoryview(quantum_state_gpu) -> Optional[memoryview]:
    """
    创建指向 GPU 数据的内存视图（实验性功能）
    
    注意：这个功能可能不安全，仅用于高级用户
    
    Args:
        quantum_state_gpu: QuantumStateGpu 实例
        
    Returns:
        memoryview: 指向 GPU 内存的视图（如果支持）
    """
    interface = GPUDataInterface(quantum_state_gpu)
    
    # 这里需要非常小心，直接访问 GPU 内存可能不安全
    # 在实际应用中，建议使用 copy_to_host 方法
    print(f"警告: 直接 GPU 内存访问是实验性功能")
    print(f"GPU 指针: 0x{interface.gpu_ptr:x}")
    
    return None  # 暂时返回 None，实际实现需要更复杂的处理


def example_usage():
    """演示 GPU 数据接口的使用"""
    
    try:
        from qulacs import QuantumStateGpu, QuantumCircuit
        from qulacs.gate import H, CNOT
        
        print("=== GPU 数据接口示例 ===\n")
        
        # 创建量子状态和电路
        n_qubits = 3
        circuit = QuantumCircuit(n_qubits)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        
        gpu_state = QuantumStateGpu(n_qubits)
        circuit.update_quantum_state(gpu_state)
        
        # 创建 GPU 数据接口
        interface = GPUDataInterface(gpu_state)
        
        # 显示指针信息
        info = interface.get_gpu_pointer_info()
        print("GPU 指针信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        if CUDA_AVAILABLE:
            print("\n--- 使用 ctypes 复制数据 ---")
            
            # 复制到主机
            host_data = interface.copy_to_host()
            print(f"复制的数据形状: {host_data.shape}")
            print(f"数据范数: {np.sum(np.abs(host_data)**2):.6f}")
            print(f"前3个元素: {host_data[:3]}")
            
            # 修改数据并复制回 GPU
            modified_data = host_data * np.exp(1j * np.pi / 4)  # 添加全局相位
            interface.copy_from_host(modified_data)
            interface.synchronize()
            
            print("\n应用全局相位后:")
            new_data = interface.copy_to_host()
            print(f"新数据范数: {np.sum(np.abs(new_data)**2):.6f}")
            
        else:
            print("CUDA Runtime 不可用，跳过数据复制演示")
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了支持 GPU 的 qulacs 版本")
    except Exception as e:
        print(f"运行错误: {e}")


if __name__ == "__main__":
    example_usage()
