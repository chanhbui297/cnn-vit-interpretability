// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_ATOM_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_ATOM_H_

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_and_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_and_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_and_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_and_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.and.b32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_xor_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_xor_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_xor_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_xor_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.xor.b32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_or_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_or_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_or_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_or_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.or.b32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .and }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_and_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_and_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .and }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_and_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_and_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.and.b64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .xor }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_xor_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_xor_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .xor }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_xor_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_xor_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.xor.b64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .or }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_or_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_or_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .or }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_or_op_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_or_op_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.or.b64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .exch }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_exch_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_exch_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_exch (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.exch.b32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .exch }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_exch_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_exch_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_exch (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.exch.b32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .cas }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_cas_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_cas_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_cas (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.cas.b32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .cas }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_cas_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B32* smem_dst,
  const B32* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_cas_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B32* __smem_dst,
  const _B32* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_cas (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.cas.b32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .exch }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_exch_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_exch_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_exch (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.exch.b64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .exch }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_exch_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_exch_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_exch (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.exch.b64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .cas }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_cas_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_cas_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_cas (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.cas.b64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .cas }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_cas_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B64* smem_dst,
  const B64* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_cas_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B64* __smem_dst,
  const _B64* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_cas (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.cas.b64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b128 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .exch }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_exch_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B128* smem_dst,
  const B128* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_exch_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B128* __smem_dst,
  const _B128* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_exch (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.exch.b128 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b128
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .exch }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_exch_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B128* smem_dst,
  const B128* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_exch_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B128* __smem_dst,
  const _B128* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_exch (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.exch.b128 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b128 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .cas }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_cas_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B128* smem_dst,
  const B128* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_cas_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B128* __smem_dst,
  const _B128* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_cas (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.cas.b128 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.b128
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .cas }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_cas_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  B128* smem_dst,
  const B128* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_cas_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  _B128* __smem_dst,
  const _B128* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_cas (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.cas.b128 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint32_t* smem_dst,
  const uint32_t* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint32_t* __smem_dst,
  const ::cuda::std::uint32_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.u32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint32_t* smem_dst,
  const uint32_t* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint32_t* __smem_dst,
  const ::cuda::std::uint32_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.min.u32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint32_t* smem_dst,
  const uint32_t* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint32_t* __smem_dst,
  const ::cuda::std::uint32_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.u32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint32_t* smem_dst,
  const uint32_t* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint32_t* __smem_dst,
  const ::cuda::std::uint32_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.max.u32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t* smem_dst,
  const uint64_t* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t* __smem_dst,
  const ::cuda::std::uint64_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.u64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t* smem_dst,
  const uint64_t* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t* __smem_dst,
  const ::cuda::std::uint64_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.min.u64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t* smem_dst,
  const uint64_t* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t* __smem_dst,
  const ::cuda::std::uint64_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.u64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t* smem_dst,
  const uint64_t* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t* __smem_dst,
  const ::cuda::std::uint64_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.max.u64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16x2 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __half2* smem_dst,
  const __half2* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __half2* __smem_dst,
  const __half2* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.f16x2 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.f16x2
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __half2* smem_dst,
  const __half2* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __half2* __smem_dst,
  const __half2* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.min.f16x2 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16x2 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __half2* smem_dst,
  const __half2* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __half2* __smem_dst,
  const __half2* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.f16x2 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.f16x2
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __half2* smem_dst,
  const __half2* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __half2* __smem_dst,
  const __half2* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.max.f16x2 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16x2 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __nv_bfloat162* smem_dst,
  const __nv_bfloat162* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __nv_bfloat162* __smem_dst,
  const __nv_bfloat162* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.bf16x2 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.bf16x2
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .min }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_min_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __nv_bfloat162* smem_dst,
  const __nv_bfloat162* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __nv_bfloat162* __smem_dst,
  const __nv_bfloat162* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.min.bf16x2 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16x2 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __nv_bfloat162* smem_dst,
  const __nv_bfloat162* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __nv_bfloat162* __smem_dst,
  const __nv_bfloat162* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.bf16x2 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.bf16x2
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .max }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_max_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __nv_bfloat162* smem_dst,
  const __nv_bfloat162* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __nv_bfloat162* __smem_dst,
  const __nv_bfloat162* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.max.bf16x2 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint32_t* smem_dst,
  const uint32_t* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint32_t* __smem_dst,
  const ::cuda::std::uint32_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.u32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint32_t* smem_dst,
  const uint32_t* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint32_t* __smem_dst,
  const ::cuda::std::uint32_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.add.u32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t* smem_dst,
  const uint64_t* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t* __smem_dst,
  const ::cuda::std::uint64_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.u64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t* smem_dst,
  const uint64_t* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t* __smem_dst,
  const ::cuda::std::uint64_t* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.add.u64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16x2 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __half2* smem_dst,
  const __half2* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __half2* __smem_dst,
  const __half2* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f16x2 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.f16x2
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __half2* smem_dst,
  const __half2* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __half2* __smem_dst,
  const __half2* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.add.f16x2 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16x2 [dstLeId,
dstDataOff], [smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __nv_bfloat162* smem_dst,
  const __nv_bfloat162* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __nv_bfloat162* __smem_dst,
  const __nv_bfloat162* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.bf16x2 "
      "[%0, %1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.bf16x2
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  __nv_bfloat162* smem_dst,
  const __nv_bfloat162* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  __nv_bfloat162* __smem_dst,
  const __nv_bfloat162* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.add.bf16x2 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f32 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  float* smem_dst,
  const float* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  float* __smem_dst,
  const float* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f32 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.f32
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  float* smem_dst,
  const float* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  float* __smem_dst,
  const float* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.add.f32 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f64 [dstLeId, dstDataOff],
[smem_dst], [smem_src], [smem_bar]; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  double* smem_dst,
  const double* smem_src,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  double* __smem_dst,
  const double* __smem_src,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f64 [%0, "
      "%1], [%2], [%3], [%4];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_atom.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.sem.scope.op.f64
[dstLeId, dstDataOff], [smem_dst], [smem_src], [smem_bar], cache_policy; // PTX ISA 94, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .op        = { .add }
template <typename = void>
__device__ void fabric_try_atom_L2_cache_hint(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::op_add_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  double* smem_dst,
  const double* smem_src,
  uint64_t* smem_bar,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE_API void fabric_try_atom_L2_cache_hint(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  double* __smem_dst,
  const double* __smem_src,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.L2::cache_hint.relaxed."
      "sys.add.f64 [%0, %1], [%2], [%3], [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__smem_dst)),
        "r"(__as_ptr_smem(__smem_src)),
        "r"(__as_ptr_smem(__smem_bar)),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_ATOM_H_
