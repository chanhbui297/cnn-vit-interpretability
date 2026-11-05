// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_GET_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_GET_H_

/*
// fabric.try_get.async.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.b128 [dstMem], [srcLeId,
srcDataOff], size, [smem_bar]; // PTX ISA 93, SM_100
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128 [%0], "
      "[%1, %2], %3, [%4];"
      :
      : "r"(__as_ptr_smem(__dstMem)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
//
fabric.try_get.async.tensor.1d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4}], [%5];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.1d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4}], [%5], %6;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.1d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.override::global_dim.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a,
SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_get_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.override::global_dim.relaxed.sys.b128 [%0], [%1, %2, %3, {%4}, {%5}], [%6];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.1d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94,
SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ void fabric_try_get_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.1d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim.relaxed.sys.b128 [%0], [%1, %2, %3, {%4}, {%5}], [%6], %7;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.2d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5}], [%6];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.2d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5}], [%6], %7;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.2d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5}, {%6}, %7, {%8, %9}], "
      "[%10];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.2d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5}, {%6}, "
      "%7, {%8, %9}], [%10], %11;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.3d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6}], [%7];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.3d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6}], [%7], %8;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.3d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6}, {%7, %8}, %9, "
      "{%10, %11, %12}], [%13];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.3d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.3d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6}, "
      "{%7, %8}, %9, {%10, %11, %12}], [%13], %14;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.4d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7}], [%8];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.4d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7}], [%8], %9;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.4d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7}, {%8, %9, %10}, "
      "%11, {%12, %13, %14, %15}], [%16];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
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
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.4d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.4d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7}, "
      "{%8, %9, %10}, %11, {%12, %13, %14, %15}], [%16], %17;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
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
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.5d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7, %8}], [%9];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.5d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ void fabric_try_get_tensor_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_get_tensor_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7, %8}], [%9], %10;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.5d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::"
      "global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7, %8}, {%9, %10, "
      "%11, %12}, %13, {%14, %15, %16, %17, %18}], [%19];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
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
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_get.async.tensor.5d.dst.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_hint.override::global_address.override::global_dim_stride.sem.scope.b128
[dstMem], [tensorMap, srcLeId, srcDataOff, tensorSizeToOverride, tensorLowerStrideToOverride,
tensorUpperStrideToOverride, tensorCoords], [smem_bar], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ void fabric_try_get_tensor_override_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  const void* tensorMap,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_get_tensor_override_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  const void* __tensorMap,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("fabric.try_get.async.tensor.5d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.L2::cache_"
      "hint.override::global_address.override::global_dim_stride.relaxed.sys.b128 [%0], [%1, %2, %3, {%4, %5, %6, %7, "
      "%8}, {%9, %10, %11, %12}, %13, {%14, %15, %16, %17, %18}], [%19], %20;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__srcLeId),
        "l"(__srcDataOff),
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
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_GET_H_
