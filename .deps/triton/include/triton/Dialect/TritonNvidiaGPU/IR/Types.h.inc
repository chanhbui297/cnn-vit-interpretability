/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton::nvidia_gpu {

/// Im2col tensor descriptor type for NVIDIA TMA operations
/// Tensor descriptor type for im2col (image-to-column) tensor memory access.
/// This is used for convolution-friendly access patterns with TMA on NVIDIA GPUs.
/// 
/// Im2col mode transforms a multi-dimensional tensor into a 2D matrix format
/// suitable for matrix multiplication, which is commonly used in convolution
/// operations.
/// 
/// Parameters:
/// - shape: The block dimensions
/// - elementType: The element type of the data block
/// - sharedLayout: Optional shared memory encoding (swizzle pattern, etc.)
/// 
/// This type implements TensorDescInterface, sharing common operations with
/// the tiled TensorDescType in the base Triton dialect.
/// 
/// See NVIDIA PTX documentation for im2col tensor mode:
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode
class TensorDescIm2ColType;
namespace detail {

struct TensorDescIm2ColTypeStorage;

} // namespace detail
class TensorDescIm2ColType : public ::mlir::Type::TypeBase<TensorDescIm2ColType, ::mlir::Type, detail::TensorDescIm2ColTypeStorage, ::mlir::triton::TensorDescInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.tensordesc_im2col";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  using Base::getChecked;
  static TensorDescIm2ColType get(llvm::ArrayRef<int64_t> shape, Type elementType, Attribute sharedLayout, bool isSigned);
  static TensorDescIm2ColType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, llvm::ArrayRef<int64_t> shape, Type elementType, Attribute sharedLayout, bool isSigned);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute sharedLayout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute sharedLayout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensordesc_im2col"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  Attribute getSharedLayout() const;
};

} // namespace mlir::triton::nvidia_gpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorDescIm2ColType)

#endif // GET_TYPEDEF_CLASSES

