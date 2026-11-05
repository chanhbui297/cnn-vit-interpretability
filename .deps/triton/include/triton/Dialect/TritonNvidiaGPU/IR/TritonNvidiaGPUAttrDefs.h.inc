/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton::nvidia_gpu {

/// NVIDIA global-memory cache hints.
/// 
/// The attribute can contain a cache modifier, an L1 eviction priority, an L2
/// fractional eviction policy, and an L2 prefetch size. This attribute is a
/// performance hint and has no memory-ordering semantics.
class CachePolicyAttr;
/// Attribute to indicate that the memory descriptor points to tensor memory.
/// The memory is laid out in blocks of size blockM x blockN. Each block is distributed
/// across TMEM 128 rows.
/// 
/// Blocks are distributed along M dimension first and then N dimension. This is an arbitrary
/// convention that needs to be followed by operations reading/writing to TMEM.
/// 
/// a tensor <128x128xf32> with blockM = 64 and blockN = 32 will be distributed as follows:
/// 
///     \ col    0        1            31         32            64            96           127
/// rows: 0  ( 0,  0) ( 0,  1) ... ( 0,  31)  ( 0,  32) ... ( 0,  64) ... ( 0,  96) ... ( 0,  127)
///       1
///      ...
///       15 (15,  0) (15,  1) ... (15,  31)  (15,  32) ... (15,  64) ... (15,  96) ... (15,  127)
///       16 (64,  0) (64,  1) ... (64,  31)  (64,  32) ... (64,  64) ... (64,  96) ... (64,  127)
///      ...
///       31 (79,  0) (79,  1) ... (79,  31)  (79,  32) ... (79,  64) ... (79,  96) ... (79,  127)
///       32 (16,  0) (16,  1) ... (16,  31)  (16,  32) ... (16,  64) ... (16,  96) ... (16,  127)
///      ..
///      127 (127, 0) (127, 1) ... (127, 31) (127, 32) ... (127, 64) ... (127, 96) ... (127, 127)
class TensorMemorySpaceAttr;
/// allowed 32-bit signless integer cases: 1, 2
class TMEMLoadReduceModifierAttr;
/// allowed 32-bit signless integer cases: 0, 1
class TensorMemoryScalesBlockRepOrderAttr;
/// An encoding to represent the different way the tensor memory is laid out.
/// `colStride` describes the stride in elements along the column dimension,
/// that is, the stride between two elements in the same row.
/// When colStride is 1 the tensor memory is packed. When colStride > 1, the
/// tensor memory between elements is undefined.
/// `twoCTAs` indicates that the tensor memory is laid out for twoCTA mode,
/// i.e., `cta_group::2`.
class TensorMemoryEncodingAttr;
/// An encoding to represent the layout of tensor memory scales.
/// As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
/// as a multiple copies of a block, each of which having the size 32x4x4B. Moreover, such blocks are duplicated
/// over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
/// layout.
/// The parameter blockRepOrder determines the ordering of repeated scale blocks along TMEM columns. Here `mn`
/// denotes the MMA instruction's non-K matrix dimension: `M` for A scales and `N` for B scales. `mnThenK`
/// advances along that dimension before advancing K, while `kThenMn` advances along K before advancing the
/// non-K dimension. When one MMA instruction consumes only one scale block along K, the value of this parameter
/// does not matter. Otherwise, it must match the hardware instruction layout. For example, Rubin NVFP4 A scales
/// with `BLOCK_M = 256` require `kThenMn`, while B scales keep the default `mnThenK`.
class TensorMemoryScalesEncodingAttr;
namespace detail {

struct CachePolicyAttrStorage;

} // namespace detail
class CachePolicyAttr : public ::mlir::Attribute::AttrBase<CachePolicyAttr, ::mlir::Attribute, detail::CachePolicyAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.cache_policy";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  using Base::getChecked;
  static CachePolicyAttr get(::mlir::MLIRContext *context, ::mlir::triton::CacheModifier cache_modifier, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l1, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_primary, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_secondary, FloatAttr l2_fraction, IntegerAttr l2_prefetch_size);
  static CachePolicyAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::CacheModifier cache_modifier, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l1, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_primary, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_secondary, FloatAttr l2_fraction, IntegerAttr l2_prefetch_size);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::CacheModifier cache_modifier, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l1, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_primary, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_secondary, FloatAttr l2_fraction, IntegerAttr l2_prefetch_size);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::CacheModifier cache_modifier, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l1, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_primary, ::mlir::triton::nvidia_gpu::CacheEvictionPriority l2_secondary, FloatAttr l2_fraction, IntegerAttr l2_prefetch_size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cache_policy"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::CacheModifier getCacheModifier() const;
  ::mlir::triton::nvidia_gpu::CacheEvictionPriority getL1() const;
  ::mlir::triton::nvidia_gpu::CacheEvictionPriority getL2Primary() const;
  ::mlir::triton::nvidia_gpu::CacheEvictionPriority getL2Secondary() const;
  FloatAttr getL2Fraction() const;
  IntegerAttr getL2PrefetchSize() const;
};
class TensorMemorySpaceAttr : public ::mlir::Attribute::AttrBase<TensorMemorySpaceAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.tensor_memory";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory"};
  }

};
namespace detail {

struct TMEMLoadReduceModifierAttrStorage;

} // namespace detail
class TMEMLoadReduceModifierAttr : public ::mlir::Attribute::AttrBase<TMEMLoadReduceModifierAttr, ::mlir::Attribute, detail::TMEMLoadReduceModifierAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.redOp";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static TMEMLoadReduceModifierAttr get(::mlir::MLIRContext *context, ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"redOp"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier getValue() const;
};
namespace detail {

struct TensorMemoryScalesBlockRepOrderAttrStorage;

} // namespace detail
class TensorMemoryScalesBlockRepOrderAttr : public ::mlir::Attribute::AttrBase<TensorMemoryScalesBlockRepOrderAttr, ::mlir::Attribute, detail::TensorMemoryScalesBlockRepOrderAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.blockRepOrder";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static TensorMemoryScalesBlockRepOrderAttr get(::mlir::MLIRContext *context, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"blockRepOrder"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder getValue() const;
};
namespace detail {

struct TensorMemoryEncodingAttrStorage;

} // namespace detail
class TensorMemoryEncodingAttr : public ::mlir::Attribute::AttrBase<TensorMemoryEncodingAttr, ::mlir::Attribute, detail::TensorMemoryEncodingAttrStorage, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.tensor_memory_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  using Base::getChecked;
  static TensorMemoryEncodingAttr get(::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, unsigned colStride, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, bool twoCTAs = false, bool fp4Padded = false);
  static TensorMemoryEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, unsigned colStride, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, bool twoCTAs = false, bool fp4Padded = false);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned blockM, unsigned blockN, unsigned colStride, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, bool twoCTAs, bool fp4Padded);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned blockM, unsigned blockN, unsigned colStride, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, bool twoCTAs, bool fp4Padded);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory_encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getBlockM() const;
  unsigned getBlockN() const;
  unsigned getColStride() const;
  ::mlir::triton::gpu::CGAEncodingAttr getCGALayout() const;
  bool getTwoCTAs() const;
  bool getFp4Padded() const;
};
namespace detail {

struct TensorMemoryScalesEncodingAttrStorage;

} // namespace detail
class TensorMemoryScalesEncodingAttr : public ::mlir::Attribute::AttrBase<TensorMemoryScalesEncodingAttr, ::mlir::Attribute, detail::TensorMemoryScalesEncodingAttrStorage, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.tensor_memory_scales_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  using Base::getChecked;
  static TensorMemoryScalesEncodingAttr get(::mlir::MLIRContext *context, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder blockRepOrder);
  static TensorMemoryScalesEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder blockRepOrder);
  static TensorMemoryScalesEncodingAttr get(::mlir::MLIRContext *context, ::mlir::triton::gpu::CGAEncodingAttr CGALayout);
  static TensorMemoryScalesEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::gpu::CGAEncodingAttr CGALayout);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder blockRepOrder);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::gpu::CGAEncodingAttr CGALayout, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder blockRepOrder);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory_scales_encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::gpu::CGAEncodingAttr getCGALayout() const;
  ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder getBlockRepOrder() const;
};

} // namespace mlir::triton::nvidia_gpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::CachePolicyAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMEMLoadReduceModifierAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrderAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr)

#endif // GET_ATTRDEF_CLASSES

