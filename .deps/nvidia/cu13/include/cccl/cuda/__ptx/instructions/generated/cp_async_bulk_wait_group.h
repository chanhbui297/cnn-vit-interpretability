// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_WAIT_GROUP_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_WAIT_GROUP_H_

/*
// cp.async.bulk.wait_group N; // PTX ISA 80, SM_90
template <int N32 = 0>
__device__ void cp_async_bulk_wait_group(
  cuda::ptx::n32_t<N32> N = {});
*/
#if __cccl_ptx_isa >= 800
template <int _N32 = 0>
_CCCL_DEVICE_API void cp_async_bulk_wait_group(::cuda::ptx::n32_t<_N32> __N = {})
{
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(__N.value) : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.wait_group.read N; // PTX ISA 80, SM_90
template <int N32 = 0>
__device__ void cp_async_bulk_wait_group_read(
  cuda::ptx::n32_t<N32> N = {});
*/
#if __cccl_ptx_isa >= 800
template <int _N32 = 0>
_CCCL_DEVICE_API void cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<_N32> __N = {})
{
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(__N.value) : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.wait_group.sync_restrict::proxy::async N; // PTX ISA 95, SM_90
template <int N32 = 0>
__device__ void cp_async_bulk_wait_group_sync_restrict_proxy_async(
  cuda::ptx::n32_t<N32> N = {});
*/
#if __cccl_ptx_isa >= 950
template <int _N32 = 0>
_CCCL_DEVICE_API void cp_async_bulk_wait_group_sync_restrict_proxy_async(::cuda::ptx::n32_t<_N32> __N = {})
{
  asm volatile("cp.async.bulk.wait_group.sync_restrict::proxy::async %0;" : : "n"(__N.value) : "memory");
}
#endif // __cccl_ptx_isa >= 950

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_WAIT_GROUP_H_
