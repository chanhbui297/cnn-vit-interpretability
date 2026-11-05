// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_PUT_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_PUT_H_

/*
// fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.b128 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 [%0, %1], "
      "[%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.sem.scope.b128 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar], bytemask; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_cp_mask(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint16_t bytemask);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_cp_mask(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint16_t __bytemask)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128 "
      "[%0, %1], [%2], %3, [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__bytemask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.b128 [dstLeId,
dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "b128 [%0, %1, %2], [%3], %4, [%5];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.b128 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
      "[%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.sem.scope.b128
[dstLeId, dstDataOff], [srcMem], size, [smem_bar], bytemask; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_multimem_cp_mask(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint16_t bytemask);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_multimem_cp_mask(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint16_t __bytemask)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed."
      "sys.b128 [%0, %1], [%2], %3, [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__bytemask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.b128
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.b128 [%0, %1, %2], [%3], %4, [%5];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0, %1, %2, {%3}], [%4], %5, [%6];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3}], [%4], %5, [%6], %7;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4}], [%5], %6, [%7];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4}], [%5], %6, [%7], %8;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3}], [%4], %5, [%6];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3}], [%4], %5, [%6], %7;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4}], [%5], %6, [%7];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4}], [%5], %6, [%7], "
      "%8;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, {%3}, {%4}], [%5], %6, [%7];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX
ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm(
    "fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
    "hint.override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, {%3}, {%4}], [%5], %6, [%7], %8;"
    :
    : "l"(__tensorMap),
      "r"(__dstLeId),
      "l"(__dstDataOff),
      "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
      "r"(__tensorCoords[0]),
      "r"(__as_ptr_smem(__srcMem)),
      "r"(__size),
      "r"(__as_ptr_smem(__smem_bar)),
      "l"(__cache_policy)
    : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX
ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm(
    "fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
    "bytes.override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, %3, {%4}, {%5}], [%6], %7, [%8];"
    :
    : "l"(__tensorMap),
      "r"(__dstLeId),
      "l"(__dstDataOff),
      "l"(__dstCounterOff),
      "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
      "r"(__tensorCoords[0]),
      "r"(__as_ptr_smem(__srcMem)),
      "r"(__size),
      "r"(__as_ptr_smem(__smem_bar))
    : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar],
cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, %3, {%4}, "
      "{%5}], [%6], %7, [%8], %9;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_multimem_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, {%3}, {%4}], [%5], %6, [%7];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX
ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, {%3}, {%4}], [%5], %6, "
      "[%7], %8;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX
ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_multimem_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, %3, {%4}, {%5}], "
      "[%6], %7, [%8];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.1d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorCoords], [srcMem], size, [smem_bar],
cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.1d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.override::global_dim.relaxed.sys.b128 [%0, %1, %2, %3, "
      "{%4}, {%5}], [%6], %7, [%8], %9;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4}], [%5], %6, [%7];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4}], [%5], %6, [%7], %8;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5}], [%6], %7, [%8];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5}], [%6], %7, [%8], %9;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4}], [%5], %6, [%7];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4}], [%5], %6, [%7], %8;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5}], [%6], %7, [%8];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5}], [%6], %7, "
      "[%8], %9;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], "
      "%10, [%11];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4}, {%5}, %6, "
      "{%7, %8}], [%9], %10, [%11], %12;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5}, {%6}, "
      "%7, {%8, %9}], [%10], %11, [%12];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, "
      "{%4, %5}, {%6}, %7, {%8, %9}], [%10], %11, [%12], %13;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4}, {%5}, %6, {%7, "
      "%8}], [%9], %10, [%11];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4}, {%5}, "
      "%6, {%7, %8}], [%9], %10, [%11], %12;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5}, "
      "{%6}, %7, {%8, %9}], [%10], %11, [%12];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.2d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, "
      "%2, %3, {%4, %5}, {%6}, %7, {%8, %9}], [%10], %11, [%12], %13;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}], [%6], %7, [%8];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}], [%6], %7, [%8], %9;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6}], [%7], %8, [%9];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6}], [%7], %8, [%9], "
      "%10;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}], [%6], %7, [%8];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}], [%6], %7, [%8], %9;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6}], [%7], %8, [%9];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6}], [%7], "
      "%8, [%9], %10;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, "
      "%11}], [%12], %13, [%14];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}, {%6, %7}, "
      "%8, {%9, %10, %11}], [%12], %13, [%14], %15;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6}, {%7, "
      "%8}, %9, {%10, %11, %12}], [%13], %14, [%15];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, "
      "{%4, %5, %6}, {%7, %8}, %9, {%10, %11, %12}], [%13], %14, [%15], %16;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}, {%6, %7}, %8, "
      "{%9, %10, %11}], [%12], %13, [%14];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5}, "
      "{%6, %7}, %8, {%9, %10, %11}], [%12], %13, [%14], %15;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, "
      "%6}, {%7, %8}, %9, {%10, %11, %12}], [%13], %14, [%15];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.3d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.3d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, "
      "%2, %3, {%4, %5, %6}, {%7, %8}, %9, {%10, %11, %12}], [%13], %14, [%15], %16;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}], [%7], %8, [%9];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}], [%7], %8, [%9], %10;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7}], [%8], %9, [%10];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7}], [%8], %9, "
      "[%10], %11;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}], [%7], %8, [%9];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}], [%7], %8, [%9], %10;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7}], [%8], %9, [%10];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7}], "
      "[%8], %9, [%10], %11;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, "
      "{%11, %12, %13, %14}], [%15], %16, [%17];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}, {%7, "
      "%8, %9}, %10, {%11, %12, %13, %14}], [%15], %16, [%17], %18;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7}, "
      "{%8, %9, %10}, %11, {%12, %13, %14, %15}], [%16], %17, [%18];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, "
      "{%4, %5, %6, %7}, {%8, %9, %10}, %11, {%12, %13, %14, %15}], [%16], %17, [%18], %19;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6}, {%7, %8, "
      "%9}, %10, {%11, %12, %13, %14}], [%15], %16, [%17];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, "
      "%6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16, [%17], %18;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, "
      "%6, %7}, {%8, %9, %10}, %11, {%12, %13, %14, %15}], [%16], %17, [%18];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.4d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.4d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, "
      "%2, %3, {%4, %5, %6, %7}, {%8, %9, %10}, %11, {%12, %13, %14, %15}], [%16], %17, [%18], %19;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}], [%8], %9, [%10];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}], [%8], %9, [%10], %11;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7, %8}], [%9], %10, [%11];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7, %8}], [%9], "
      "%10, [%11], %12;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}], [%8], %9, [%10];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}], [%8], %9, [%10], %11;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm(
    "fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
    "counted::bytes.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7, %8}], [%9], %10, [%11];"
    :
    : "l"(__tensorMap),
      "r"(__dstLeId),
      "l"(__dstDataOff),
      "l"(__dstCounterOff),
      "r"(__tensorCoords[0]),
      "r"(__tensorCoords[1]),
      "r"(__tensorCoords[2]),
      "r"(__tensorCoords[3]),
      "r"(__tensorCoords[4]),
      "r"(__as_ptr_smem(__srcMem)),
      "r"(__size),
      "r"(__as_ptr_smem(__smem_bar))
    : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7, %8}], "
      "[%9], %10, [%11], %12;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, "
      "%11}, %12, {%13, %14, %15, %16, %17}], [%18], %19, [%20];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19, [%20], %21;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, %6, %7, "
      "%8}, {%9, %10, %11, %12}, %13, {%14, %15, %16, %17, %18}], [%19], %20, [%21];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::"
      "bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, "
      "{%4, %5, %6, %7, %8}, {%9, %10, %11, %12}, %13, {%14, %15, %16, %17, %18}], [%19], %20, [%21], %22;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, %7}, {%8, "
      "%9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19, [%20];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2:"
      ":cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, {%3, %4, %5, %6, "
      "%7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19, [%20], %21;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, %2, %3, {%4, %5, "
      "%6, %7, %8}, {%9, %10, %11, %12}, %13, {%14, %15, %16, %17, %18}], [%19], %20, [%21];"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_put.async.tensor.multimem.5d.src.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[tensorMap, dstLeId, dstDataOff, dstCounterOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [srcMem], size, [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  const void* tensorMap,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_put_tensor_multimem_counted_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  const void* __tensorMap,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_put.async.tensor.multimem.5d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric."
      "counted::bytes.L2::cache_hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0, %1, "
      "%2, %3, {%4, %5, %6, %7, %8}, {%9, %10, %11, %12}, %13, {%14, %15, %16, %17, %18}], [%19], %20, [%21], %22;"
      :
      : "l"(__tensorMap),
        "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_PUT_H_
