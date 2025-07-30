#pragma once

#ifdef _USE_GPU

#include <cstdint>
#include "state_gpu.hpp"

namespace qulacs {
namespace gpu_interop {

/**
 * GPU 数据交互工具类
 */
class GPUDataAccessor {
public:
    /**
     * 获取 QuantumStateGpu 的 GPU 数据指针
     */
    static uintptr_t get_gpu_data_ptr(const QuantumStateGpu* state) {
        return reinterpret_cast<uintptr_t>(state->data());
    }
    
    /**
     * 获取 GPU 设备号
     */
    static unsigned int get_device_number(const QuantumStateGpu* state) {
        return state->device_number;
    }
    
    /**
     * 获取状态维度
     */
    static ITYPE get_dimension(const QuantumStateGpu* state) {
        return state->dim;
    }
    
    /**
     * 创建指向外部 GPU 内存的 QuantumStateGpu
     * 注意：这是一个危险的操作，需要确保外部内存的生命周期
     */
    static QuantumStateGpu* create_from_gpu_ptr(
        UINT qubit_count, 
        uintptr_t gpu_ptr, 
        unsigned int device_number
    ) {
        // 这里需要特殊处理，创建一个不拥有内存的 QuantumStateGpu
        // 实际实现可能需要修改 QuantumStateGpu 的构造函数
        throw std::runtime_error("create_from_gpu_ptr not implemented yet");
    }
    
    /**
     * 同步 GPU 操作
     */
    static void synchronize_device(unsigned int device_number) {
        // 调用 CUDA 同步函数
        set_device(device_number);
        gpuDeviceSynchronize();
    }
};

} // namespace gpu_interop
} // namespace qulacs

#endif // _USE_GPU
