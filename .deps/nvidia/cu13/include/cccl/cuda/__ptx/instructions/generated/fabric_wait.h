// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_WAIT_H_
#define _CUDA_PTX_GENERATED_FABRIC_WAIT_H_

/*
// fabric.wait.sync_restrict::reads; // PTX ISA 93-94, SM_100
template <typename = void>
__device__ void fabric_wait();
*/
#if __cccl_ptx_isa >= 930 && __cccl_ptx_isa < 950
template <typename = void>
_CCCL_DEVICE_API void fabric_wait()
{
  asm volatile("fabric.wait.sync_restrict::reads;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 930 && __cccl_ptx_isa < 950

/*
// fabric.wait.sync_restrict::read N; // PTX ISA 95, SM_100
template <int N32 = 0>
__device__ void fabric_wait(
  cuda::ptx::n32_t<N32> N = {});
*/
#if __cccl_ptx_isa >= 950
template <int _N32 = 0>
_CCCL_DEVICE_API void fabric_wait(::cuda::ptx::n32_t<_N32> __N = {})
{
  asm volatile("fabric.wait.sync_restrict::read %0;" : : "n"(__N.value) : "memory");
}
#endif // __cccl_ptx_isa >= 950

#endif // _CUDA_PTX_GENERATED_FABRIC_WAIT_H_
