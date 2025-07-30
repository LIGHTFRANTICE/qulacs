#!/usr/bin/env python3
"""
示例：直接访问 QuantumStateGpu 中的 GPU 数据

这个示例展示了如何在 Python 程序中直接访问存储在 GPU 中的量子状态数据，
而无需将数据复制到 CPU 内存。
"""

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. GPU data access will be limited.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("Warning: PyCUDA not available. GPU data access will be limited.")

from qulacs import QuantumStateGpu, QuantumCircuit
from qulacs.gate import H, CNOT, RY


class GPUQuantumStateAccessor:
    """
    用于直接访问 QuantumStateGpu GPU 数据的工具类
    """
    
    def __init__(self, quantum_state_gpu):
        """
        初始化 GPU 数据访问器
        
        Args:
            quantum_state_gpu: QuantumStateGpu 实例
        """
        if not hasattr(quantum_state_gpu, 'get_gpu_ptr'):
            raise RuntimeError("QuantumStateGpu does not support direct GPU data access. "
                             "Please update your qulacs installation.")
        
        self.qstate = quantum_state_gpu
        self.gpu_ptr = quantum_state_gpu.get_gpu_ptr()
        self.device_number = quantum_state_gpu.get_device_number()
        self.dim = quantum_state_gpu.get_qubit_count() ** 2
        self.qubit_count = quantum_state_gpu.get_qubit_count()
        
    def get_cupy_array(self, read_only=True):
        """
        使用 CuPy 直接访问 GPU 数据
        
        Args:
            read_only: 如果为 True，返回只读数组；如果为 False，返回可写的副本
        
        Returns:
            cupy.ndarray: 指向 GPU 数据的 CuPy 数组
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available")
        
        # 设置正确的 GPU 设备
        cp.cuda.Device(self.device_number).use()
        
        if read_only:
            # 创建只读的指向现有 GPU 内存的 CuPy 数组
            gpu_array = cp.ndarray(
                shape=(self.dim,),
                dtype=cp.complex128,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(self.gpu_ptr, self.dim * 16, self),
                    0
                )
            )
            gpu_array.flags.writeable = False
            return gpu_array
        else:
            # 创建数据的副本以允许修改
            readonly_array = self.get_cupy_array(read_only=True)
            return cp.copy(readonly_array)
    
    def get_pycuda_array(self):
        """
        使用 PyCUDA 直接访问 GPU 数据
        
        Returns:
            pycuda.gpuarray.GPUArray: 指向 GPU 数据的 PyCUDA 数组
        """
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA is not available")
        
        # 创建指向现有 GPU 内存的 GPUArray
        gpu_array = GPUArray(
            shape=(self.dim,),
            dtype=np.complex128,
            gpudata=cuda.mem_alloc_from_host_pointer(self.gpu_ptr)
        )
        return gpu_array
    
    def compute_gpu_statistics(self):
        """
        直接在 GPU 上计算统计信息
        
        Returns:
            dict: 包含各种统计信息的字典
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU statistics")
        
        gpu_array = self.get_cupy_array()
        
        # 在 GPU 上直接计算统计信息
        stats = {
            'norm_squared': float(cp.sum(cp.abs(gpu_array) ** 2)),
            'entropy': self._compute_entropy_gpu(gpu_array),
            'max_amplitude': float(cp.max(cp.abs(gpu_array))),
            'mean_amplitude': float(cp.mean(cp.abs(gpu_array)))
        }
        
        return stats
    
    def _compute_entropy_gpu(self, gpu_array):
        """在 GPU 上计算冯诺依曼熵"""
        probs = cp.abs(gpu_array) ** 2
        # 避免 log(0)
        probs = probs[probs > 1e-12]
        entropy = -cp.sum(probs * cp.log2(probs))
        return float(entropy)
    
    def apply_custom_gpu_operation(self, operation_func):
        """
        在 GPU 数据上应用自定义操作
        
        Args:
            operation_func: 接受 CuPy 数组并返回修改后数组的函数
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for custom GPU operations")
        
        gpu_array = self.get_cupy_array(read_only=False)  # 获取可写副本
        result = operation_func(gpu_array)
        
        # 将结果写回原始 GPU 内存
        self.update_gpu_memory(result)
    
    def update_gpu_memory(self, new_data):
        """
        安全地更新 GPU 内存
        
        Args:
            new_data: 新的数据（CuPy 数组或 numpy 数组）
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU memory updates")
        
        # 确保数据是 CuPy 数组且在正确的设备上
        if isinstance(new_data, cp.ndarray):
            if new_data.device.id != self.device_number:
                new_data = cp.asarray(new_data, device=self.device_number)
        else:
            # 如果是 numpy 数组，转换为 CuPy 数组
            cp.cuda.Device(self.device_number).use()
            new_data = cp.asarray(new_data, dtype=cp.complex128)
        
        # 检查形状和数据类型
        if new_data.shape != (self.dim,):
            raise ValueError(f"Data shape {new_data.shape} doesn't match expected {(self.dim,)}")
        if new_data.dtype != cp.complex128:
            new_data = new_data.astype(cp.complex128)
        
        # 使用 CUDA 内存复制
        cp.cuda.runtime.memcpy(
            dst=self.gpu_ptr,
            src=new_data.data.ptr,
            size=self.dim * 16,  # complex128 = 16 bytes
            kind=cp.cuda.runtime.memcpyDeviceToDevice
        )
    
    def synchronize(self):
        """同步 GPU 操作"""
        if CUPY_AVAILABLE:
            cp.cuda.Device(self.device_number).synchronize()


def example_direct_gpu_access():
    """演示直接 GPU 数据访问的示例"""
    
    print("=== 直接 GPU 数据访问示例 ===\n")
    
    # 创建量子电路和 GPU 状态
    n_qubits = 4
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_RY_gate(1, np.pi/4)
    circuit.add_CNOT_gate(1, 2)
    
    # 使用 GPU 状态
    gpu_state = QuantumStateGpu(n_qubits)
    circuit.update_quantum_state(gpu_state)
    
    print(f"量子比特数: {n_qubits}")
    print(f"状态维度: {2**n_qubits}")
    print(f"GPU 设备号: {gpu_state.get_device_number()}")
    print(f"GPU 数据指针: 0x{gpu_state.get_gpu_ptr():x}")
    
    # 创建 GPU 数据访问器
    accessor = GPUQuantumStateAccessor(gpu_state)
    
    if CUPY_AVAILABLE:
        print("\n--- 使用 CuPy 直接访问 GPU 数据 ---")
        
        # 获取 CuPy 数组
        gpu_array = accessor.get_cupy_array()
        print(f"CuPy 数组形状: {gpu_array.shape}")
        print(f"CuPy 数组数据类型: {gpu_array.dtype}")
        
        # 计算统计信息
        stats = accessor.compute_gpu_statistics()
        print(f"状态范数平方: {stats['norm_squared']:.6f}")
        print(f"熵: {stats['entropy']:.6f}")
        print(f"最大振幅: {stats['max_amplitude']:.6f}")
        print(f"平均振幅: {stats['mean_amplitude']:.6f}")
        
        # 应用自定义 GPU 操作
        def phase_shift(arr):
            """应用全局相位"""
            return arr * cp.exp(1j * cp.pi / 4)
        
        print("\n应用全局相位变换...")
        accessor.apply_custom_gpu_operation(phase_shift)
        
        # 验证操作结果
        new_stats = accessor.compute_gpu_statistics()
        print(f"变换后状态范数平方: {new_stats['norm_squared']:.6f}")
    
    # 与传统方法比较
    print("\n--- 性能比较 ---")
    import time
    
    # 传统方法：复制到 CPU
    start_time = time.time()
    cpu_data = gpu_state.get_vector()
    cpu_norm = np.sum(np.abs(cpu_data) ** 2)
    cpu_time = time.time() - start_time
    print(f"CPU 方法时间: {cpu_time:.6f} 秒")
    print(f"CPU 计算的范数: {cpu_norm:.6f}")
    
    if CUPY_AVAILABLE:
        # GPU 方法：直接访问
        start_time = time.time()
        gpu_norm = accessor.compute_gpu_statistics()['norm_squared']
        gpu_time = time.time() - start_time
        print(f"GPU 方法时间: {gpu_time:.6f} 秒")
        print(f"GPU 计算的范数: {gpu_norm:.6f}")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")


def example_interoperability():
    """演示与其他 GPU 库的互操作性"""
    
    if not CUPY_AVAILABLE:
        print("跳过互操作性示例（需要 CuPy）")
        return
    
    print("\n=== GPU 库互操作性示例 ===\n")
    
    # 创建量子状态
    n_qubits = 3
    gpu_state = QuantumStateGpu(n_qubits)
    gpu_state.set_Haar_random_state()
    
    accessor = GPUQuantumStateAccessor(gpu_state)
    qulacs_array = accessor.get_cupy_array(read_only=True)  # 获取只读数组
    
    print(f"Qulacs GPU 数据形状: {qulacs_array.shape}")
    
    # 与其他 CuPy 数组进行运算
    other_array = cp.random.random(qulacs_array.shape) + 1j * cp.random.random(qulacs_array.shape)
    other_array = other_array / cp.linalg.norm(other_array)
    
    # 计算内积
    inner_product = cp.vdot(qulacs_array, other_array)
    print(f"与随机状态的内积: {inner_product}")
    
    # 线性组合（在副本上操作）
    combined_state = 0.6 * qulacs_array + 0.8 * other_array
    combined_state = combined_state / cp.linalg.norm(combined_state)
    
    # 将结果安全地写回 Qulacs 状态
    accessor.update_gpu_memory(combined_state)
    accessor.synchronize()
    
    print("成功更新 Qulacs 量子状态")
    print(f"新状态范数: {gpu_state.get_squared_norm():.6f}")


if __name__ == "__main__":
    try:
        example_direct_gpu_access()
        example_interoperability()
    except Exception as e:
        print(f"错误: {e}")
        print("\n确保你已经:")
        print("1. 使用支持 GPU 的 qulacs 版本")
        print("2. 安装了 CuPy: pip install cupy")
        print("3. 有可用的 CUDA GPU")
