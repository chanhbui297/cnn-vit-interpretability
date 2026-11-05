/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton {

/// Common interface for tensor descriptor types.
/// 
/// This interface provides a unified API for different tensor descriptor
/// implementations (e.g., tiled TensorDescType, im2col TensorDescIm2ColType).
/// All tensor descriptors share the concept of a "block type" which describes
/// the shape and element type of the data block being accessed.
/// 
/// Concrete implementations:
/// - TensorDescType (Triton dialect): Basic tiled tensor descriptor
/// - TensorDescIm2ColType (TritonNvidiaGPU dialect): Im2col tensor descriptor
///   with additional convolution parameters
class TensorDescInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface for tensor descriptor types.
/// 
/// This interface provides a unified API for different tensor descriptor
/// implementations (e.g., tiled TensorDescType, im2col TensorDescIm2ColType).
/// All tensor descriptors share the concept of a "block type" which describes
/// the shape and element type of the data block being accessed.
/// 
/// Concrete implementations:
/// - TensorDescType (Triton dialect): Basic tiled tensor descriptor
/// - TensorDescIm2ColType (TritonNvidiaGPU dialect): Im2col tensor descriptor
///   with additional convolution parameters
namespace detail {
struct TensorDescInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    llvm::ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Type );
    mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
    mlir::Attribute (*getSharedLayout)(const Concept *impl, ::mlir::Type );
    mlir::RankedTensorType (*getBlockType)(const Concept *impl, ::mlir::Type );
    mlir::RankedTensorType (*getSignlessBlockType)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::TensorDescInterface;
    Model() : Concept{getShape, getElementType, getSharedLayout, getBlockType, getSignlessBlockType} {}

    static inline llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::Attribute getSharedLayout(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::RankedTensorType getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::RankedTensorType getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::TensorDescInterface;
    FallbackModel() : Concept{getShape, getElementType, getSharedLayout, getBlockType, getSignlessBlockType} {}

    static inline llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::Attribute getSharedLayout(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::RankedTensorType getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::RankedTensorType getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    mlir::RankedTensorType getBlockType(::mlir::Type tablegen_opaque_val) const;
    mlir::RankedTensorType getSignlessBlockType(::mlir::Type tablegen_opaque_val) const;
  };
};
template <typename ConcreteType>
struct TensorDescInterfaceTrait;

} // namespace detail
class TensorDescInterface : public ::mlir::TypeInterface<TensorDescInterface, detail::TensorDescInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<TensorDescInterface, detail::TensorDescInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::TensorDescInterfaceTrait<ConcreteType> {};

  /// Returns the shape of the descriptor block
  llvm::ArrayRef<int64_t> getShape() const;

  /// Returns the element type of the descriptor block
  mlir::Type getElementType() const;

  /// Returns the optional shared memory layout encoding
  mlir::Attribute getSharedLayout() const;

  /// Returns a block tensor type constructed from shape and element type
  mlir::RankedTensorType getBlockType() const;

  /// Returns a block tensor type constructed with signless integer element type
  mlir::RankedTensorType getSignlessBlockType() const;
};

} // namespace mlir::triton
namespace mlir::triton::detail {

  template <typename ConcreteType>
  struct TensorDescInterfaceTrait : public ::mlir::TypeInterface<TensorDescInterface, detail::TensorDescInterfaceInterfaceTraits>::Trait<ConcreteType> {

    /// Returns a block tensor type constructed from shape and element type
    mlir::RankedTensorType getBlockType() const {
      return mlir::RankedTensorType::get((*static_cast<const ConcreteType *>(this)).getShape(),
                                           (*static_cast<const ConcreteType *>(this)).getElementType());
    }

    /// Returns a block tensor type constructed with signless integer element type
    mlir::RankedTensorType getSignlessBlockType() const {
      auto shape = (*static_cast<const ConcreteType *>(this)).getShape();
        auto elemTy = (*static_cast<const ConcreteType *>(this)).getElementType();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(elemTy)) {
          auto width = intTy.getWidth();
          elemTy = mlir::IntegerType::get((*static_cast<const ConcreteType *>(this)).getContext(), width);
        }
        return mlir::RankedTensorType::get(shape, elemTy);
    }
  };

} // namespace mlir::triton::detail
namespace mlir::triton {

template<typename ConcreteType>
llvm::ArrayRef<int64_t> detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getShape();
}
template<typename ConcreteType>
mlir::Type detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
}
template<typename ConcreteType>
mlir::Attribute detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getSharedLayout(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSharedLayout();
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getBlockType();
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSignlessBlockType();
}
template<typename ConcreteType>
llvm::ArrayRef<int64_t> detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getShape(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::Type detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::Attribute detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getSharedLayout(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getSharedLayout(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getBlockType(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getSignlessBlockType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getBlockType(::mlir::Type tablegen_opaque_val) const {
return mlir::RankedTensorType::get((::llvm::cast<ConcreteType>(tablegen_opaque_val)).getShape(),
                                           (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType());
}
template<typename ConcreteModel, typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getSignlessBlockType(::mlir::Type tablegen_opaque_val) const {
auto shape = (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getShape();
        auto elemTy = (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(elemTy)) {
          auto width = intTy.getWidth();
          elemTy = mlir::IntegerType::get((::llvm::cast<ConcreteType>(tablegen_opaque_val)).getContext(), width);
        }
        return mlir::RankedTensorType::get(shape, elemTy);
}

} // namespace mlir::triton
