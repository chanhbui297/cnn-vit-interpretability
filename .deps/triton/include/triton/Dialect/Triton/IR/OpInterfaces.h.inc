/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton {

/// This interface is implemented by operations that perform a transpose.
/// It provides methods to access common properties such as the order attribute
/// and the source operand.
class TransposeOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// This interface is implemented by operations that perform a dot product.
class DotOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface to get the descriptor argument from an operation on tensor descriptors.
class DescriptorOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

class DescriptorStoreLikeOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Common marker interface for operations that load from tensor descriptors.
class DescriptorLoadLikeOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface for operations that carry a predicate or mask operand that
/// can be combined with a pipeline predicate.
/// 
/// A false predicate can still leave effects such as masked-copy zero filling
/// or MMA completion signals. Erasing an inactive operation requires a separate
/// guarantee that its predicate suppresses all effects.
class PredicatedOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Operations that can pin a compiler-chosen result layout.
/// 
/// `getEfficientLayout` is a hint that the current result encoding should be
/// kept. Layout conversion treats the operation as a layout anchor and will
/// not rewrite or rematerialize it in a different encoding. When false, the
/// operation participates in layout propagation normally.
class EfficientLayoutOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface for atomic operations with memory-ordering semantics.
class AtomicOpInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// This interface is implemented by operations that perform a transpose.
/// It provides methods to access common properties such as the order attribute
/// and the source operand.
namespace detail {
struct TransposeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayRef<int32_t> (*getOrder)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::TransposeOpInterface;
    Model() : Concept{getSrc, getOrder} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayRef<int32_t> getOrder(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::TransposeOpInterface;
    FallbackModel() : Concept{getSrc, getOrder} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayRef<int32_t> getOrder(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct TransposeOpInterfaceTrait;

} // namespace detail
class TransposeOpInterface : public ::mlir::OpInterface<TransposeOpInterface, detail::TransposeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TransposeOpInterface, detail::TransposeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TransposeOpInterfaceTrait<ConcreteOp> {};

  /// Get the source operand of the transposition.
  ::mlir::Value getSrc();

  /// Get the order of the transposition.
  ::mlir::ArrayRef<int32_t> getOrder();
};

} // namespace mlir::triton
namespace mlir::triton {

/// This interface is implemented by operations that perform a dot product.
namespace detail {
struct DotOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getA)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getB)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getD)(const Concept *impl, ::mlir::Operation *);
    bool (*verifyDims)(const Concept *impl, ::mlir::Operation *);
    bool (*verifyOutputDims)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::DotOpInterface;
    Model() : Concept{getA, getB, getD, verifyDims, verifyOutputDims} {}

    static inline ::mlir::Value getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getD(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool verifyDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool verifyOutputDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::DotOpInterface;
    FallbackModel() : Concept{getA, getB, getD, verifyDims, verifyOutputDims} {}

    static inline ::mlir::Value getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getD(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool verifyDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool verifyOutputDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool verifyDims(::mlir::Operation *tablegen_opaque_val) const;
    bool verifyOutputDims(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct DotOpInterfaceTrait;

} // namespace detail
class DotOpInterface : public ::mlir::OpInterface<DotOpInterface, detail::DotOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DotOpInterface, detail::DotOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DotOpInterfaceTrait<ConcreteOp> {};

  /// Get the LHS A tensor
  ::mlir::Value getA();

  /// Get the RHS B tensor
  ::mlir::Value getB();

  /// Get the output tensor
  ::mlir::Value getD();

  /// Verify the dimensions of the A and B DotOp operands.
  bool verifyDims();

  /// Verify the dimensions of the DotOp output.
  bool verifyOutputDims();
};

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface to get the descriptor argument from an operation on tensor descriptors.
namespace detail {
struct DescriptorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<mlir::triton::TensorDescType> (*getDesc)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::DescriptorOpInterface;
    Model() : Concept{getDesc} {}

    static inline ::mlir::TypedValue<mlir::triton::TensorDescType> getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::DescriptorOpInterface;
    FallbackModel() : Concept{getDesc} {}

    static inline ::mlir::TypedValue<mlir::triton::TensorDescType> getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DescriptorOpInterfaceTrait;

} // namespace detail
class DescriptorOpInterface : public ::mlir::OpInterface<DescriptorOpInterface, detail::DescriptorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DescriptorOpInterface, detail::DescriptorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DescriptorOpInterfaceTrait<ConcreteOp> {};

  /// Get the descriptor
  ::mlir::TypedValue<mlir::triton::TensorDescType> getDesc();
};

} // namespace mlir::triton
namespace mlir::triton {

namespace detail {
struct DescriptorStoreLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<mlir::RankedTensorType> (*getSrc)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand&(*getSrcMutable)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::triton::DescriptorOpInterface::Concept *implDescriptorOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDescriptorOpInterface = interfaceMap.lookup<::mlir::triton::DescriptorOpInterface>();
      assert(implDescriptorOpInterface && "`::mlir::triton::DescriptorStoreLikeOpInterface` expected its base interface `::mlir::triton::DescriptorOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::DescriptorStoreLikeOpInterface;
    Model() : Concept{getSrc, getSrcMutable} {}

    static inline ::mlir::TypedValue<mlir::RankedTensorType> getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand&getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::DescriptorStoreLikeOpInterface;
    FallbackModel() : Concept{getSrc, getSrcMutable} {}

    static inline ::mlir::TypedValue<mlir::RankedTensorType> getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand&getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DescriptorStoreLikeOpInterfaceTrait;

} // namespace detail
class DescriptorStoreLikeOpInterface : public ::mlir::OpInterface<DescriptorStoreLikeOpInterface, detail::DescriptorStoreLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DescriptorStoreLikeOpInterface, detail::DescriptorStoreLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DescriptorStoreLikeOpInterfaceTrait<ConcreteOp> {};

  /// Get Source tensor
  ::mlir::TypedValue<mlir::RankedTensorType> getSrc();

  /// Get mutable source tensor
  ::mlir::OpOperand&getSrcMutable();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::triton::DescriptorOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::triton::DescriptorOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::triton::DescriptorOpInterface(*this, getImpl()->implDescriptorOpInterface);
  }


  /// Get the descriptor
  ::mlir::TypedValue<mlir::triton::TensorDescType> getDesc();
};

} // namespace mlir::triton
namespace mlir::triton {

/// Common marker interface for operations that load from tensor descriptors.
namespace detail {
struct DescriptorLoadLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::triton::DescriptorOpInterface::Concept *implDescriptorOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDescriptorOpInterface = interfaceMap.lookup<::mlir::triton::DescriptorOpInterface>();
      assert(implDescriptorOpInterface && "`::mlir::triton::DescriptorLoadLikeOpInterface` expected its base interface `::mlir::triton::DescriptorOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::DescriptorLoadLikeOpInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::DescriptorLoadLikeOpInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DescriptorLoadLikeOpInterfaceTrait;

} // namespace detail
class DescriptorLoadLikeOpInterface : public ::mlir::OpInterface<DescriptorLoadLikeOpInterface, detail::DescriptorLoadLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DescriptorLoadLikeOpInterface, detail::DescriptorLoadLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DescriptorLoadLikeOpInterfaceTrait<ConcreteOp> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::triton::DescriptorOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::triton::DescriptorOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::triton::DescriptorOpInterface(*this, getImpl()->implDescriptorOpInterface);
  }


  /// Get the descriptor
  ::mlir::TypedValue<mlir::triton::TensorDescType> getDesc();
};

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface for operations that carry a predicate or mask operand that
/// can be combined with a pipeline predicate.
/// 
/// A false predicate can still leave effects such as masked-copy zero filling
/// or MMA completion signals. Erasing an inactive operation requires a separate
/// guarantee that its predicate suppresses all effects.
namespace detail {
struct PredicatedOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getPredicateOperand)(const Concept *impl, ::mlir::Operation *);
    void (*setPredicateOperand)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::Type (*getPredicateOperandTypeLike)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::PredicatedOpInterface;
    Model() : Concept{getPredicateOperand, setPredicateOperand, getPredicateOperandTypeLike} {}

    static inline ::mlir::Value getPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred);
    static inline ::mlir::Type getPredicateOperandTypeLike(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::PredicatedOpInterface;
    FallbackModel() : Concept{getPredicateOperand, setPredicateOperand, getPredicateOperandTypeLike} {}

    static inline ::mlir::Value getPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred);
    static inline ::mlir::Type getPredicateOperandTypeLike(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct PredicatedOpInterfaceTrait;

} // namespace detail
class PredicatedOpInterface : public ::mlir::OpInterface<PredicatedOpInterface, detail::PredicatedOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PredicatedOpInterface, detail::PredicatedOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PredicatedOpInterfaceTrait<ConcreteOp> {};

  /// Return the current predicate or mask operand.
  ::mlir::Value getPredicateOperand();

  /// Update the predicate or mask operand.
  void setPredicateOperand(::mlir::Value pred);

  /// Return a type whose shape determines the predicate operand type.
  ::mlir::Type getPredicateOperandTypeLike();
};

} // namespace mlir::triton
namespace mlir::triton {

/// Operations that can pin a compiler-chosen result layout.
/// 
/// `getEfficientLayout` is a hint that the current result encoding should be
/// kept. Layout conversion treats the operation as a layout anchor and will
/// not rewrite or rematerialize it in a different encoding. When false, the
/// operation participates in layout propagation normally.
namespace detail {
struct EfficientLayoutOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*getEfficientLayout)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::EfficientLayoutOpInterface;
    Model() : Concept{getEfficientLayout} {}

    static inline bool getEfficientLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::EfficientLayoutOpInterface;
    FallbackModel() : Concept{getEfficientLayout} {}

    static inline bool getEfficientLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct EfficientLayoutOpInterfaceTrait;

} // namespace detail
class EfficientLayoutOpInterface : public ::mlir::OpInterface<EfficientLayoutOpInterface, detail::EfficientLayoutOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<EfficientLayoutOpInterface, detail::EfficientLayoutOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::EfficientLayoutOpInterfaceTrait<ConcreteOp> {};

  /// Return true if the compiler has pinned the current result layout.
  bool getEfficientLayout();
};

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface for atomic operations with memory-ordering semantics.
namespace detail {
struct AtomicOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getPtr)(const Concept *impl, ::mlir::Operation *);
    ::mlir::triton::MemSemantic (*getMemSemantic)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::AtomicOpInterface;
    Model() : Concept{getPtr, getMemSemantic} {}

    static inline ::mlir::Value getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::triton::MemSemantic getMemSemantic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::AtomicOpInterface;
    FallbackModel() : Concept{getPtr, getMemSemantic} {}

    static inline ::mlir::Value getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::triton::MemSemantic getMemSemantic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct AtomicOpInterfaceTrait;

} // namespace detail
class AtomicOpInterface : public ::mlir::OpInterface<AtomicOpInterface, detail::AtomicOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AtomicOpInterface, detail::AtomicOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AtomicOpInterfaceTrait<ConcreteOp> {};

  /// Return the pointer operand accessed by the atomic operation.
  ::mlir::Value getPtr();

  /// Return the atomic memory-ordering semantic.
  ::mlir::triton::MemSemantic getMemSemantic();
};

} // namespace mlir::triton
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct TransposeOpInterfaceTrait : public ::mlir::OpInterface<TransposeOpInterface, detail::TransposeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::triton::impl::verifyTransposeOpInterface(op);
    }
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct DotOpInterfaceTrait : public ::mlir::OpInterface<DotOpInterface, detail::DotOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {

    /// Verify the dimensions of the A and B DotOp operands.
    bool verifyDims() {
      auto aShape = cast<ShapedType>((*static_cast<ConcreteOp *>(this)).getA().getType()).getShape();
        auto bShape = cast<ShapedType>((*static_cast<ConcreteOp *>(this)).getB().getType()).getShape();
        return aShape.back() == bShape[bShape.size() - 2];
    }

    /// Verify the dimensions of the DotOp output.
    bool verifyOutputDims() {
      auto aTy = cast<ShapedType>((*static_cast<ConcreteOp *>(this)).getA().getType());
        auto bTy = cast<ShapedType>((*static_cast<ConcreteOp *>(this)).getB().getType());
        auto cTy = cast<ShapedType>((*static_cast<ConcreteOp *>(this))->getOperand(2).getType());
        auto aShape = aTy.getShape();
        auto bShape = bTy.getShape();
        auto cShape = cTy.getShape();
        return cShape[cShape.size() - 2] == aShape[aShape.size() - 2] &&
               cShape[cShape.size() - 1] == bShape[aShape.size() - 1];
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::triton::impl::verifyDotOpInterface(op);
    }
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct DescriptorOpInterfaceTrait : public ::mlir::OpInterface<DescriptorOpInterface, detail::DescriptorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct DescriptorStoreLikeOpInterfaceTrait : public ::mlir::OpInterface<DescriptorStoreLikeOpInterface, detail::DescriptorStoreLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct DescriptorLoadLikeOpInterfaceTrait : public ::mlir::OpInterface<DescriptorLoadLikeOpInterface, detail::DescriptorLoadLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct PredicatedOpInterfaceTrait : public ::mlir::OpInterface<PredicatedOpInterface, detail::PredicatedOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct EfficientLayoutOpInterfaceTrait : public ::mlir::OpInterface<EfficientLayoutOpInterface, detail::EfficientLayoutOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::detail
namespace mlir::triton::detail {

  template <typename ConcreteOp>
  struct AtomicOpInterfaceTrait : public ::mlir::OpInterface<AtomicOpInterface, detail::AtomicOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::detail
namespace mlir::triton {

template<typename ConcreteOp>
::mlir::Value detail::TransposeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::ArrayRef<int32_t> detail::TransposeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOrder(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOrder();
}
template<typename ConcreteOp>
::mlir::Value detail::TransposeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayRef<int32_t> detail::TransposeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOrder(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOrder(tablegen_opaque_val);
}

} // namespace mlir::triton
namespace mlir::triton {

template<typename ConcreteOp>
::mlir::Value detail::DotOpInterfaceInterfaceTraits::Model<ConcreteOp>::getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getA();
}
template<typename ConcreteOp>
::mlir::Value detail::DotOpInterfaceInterfaceTraits::Model<ConcreteOp>::getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getB();
}
template<typename ConcreteOp>
::mlir::Value detail::DotOpInterfaceInterfaceTraits::Model<ConcreteOp>::getD(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getD();
}
template<typename ConcreteOp>
bool detail::DotOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyDims();
}
template<typename ConcreteOp>
bool detail::DotOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyOutputDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyOutputDims();
}
template<typename ConcreteOp>
::mlir::Value detail::DotOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getA(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::DotOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getB(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::DotOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getD(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getD(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::DotOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyDims(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::DotOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyOutputDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyOutputDims(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DotOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyDims(::mlir::Operation *tablegen_opaque_val) const {
auto aShape = cast<ShapedType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getA().getType()).getShape();
        auto bShape = cast<ShapedType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getB().getType()).getShape();
        return aShape.back() == bShape[bShape.size() - 2];
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DotOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyOutputDims(::mlir::Operation *tablegen_opaque_val) const {
auto aTy = cast<ShapedType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getA().getType());
        auto bTy = cast<ShapedType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getB().getType());
        auto cTy = cast<ShapedType>((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOperand(2).getType());
        auto aShape = aTy.getShape();
        auto bShape = bTy.getShape();
        auto cShape = cTy.getShape();
        return cShape[cShape.size() - 2] == aShape[aShape.size() - 2] &&
               cShape[cShape.size() - 1] == bShape[aShape.size() - 1];
}

} // namespace mlir::triton
namespace mlir::triton {

template<typename ConcreteOp>
::mlir::TypedValue<mlir::triton::TensorDescType> detail::DescriptorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDesc();
}
template<typename ConcreteOp>
::mlir::TypedValue<mlir::triton::TensorDescType> detail::DescriptorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDesc(tablegen_opaque_val);
}

} // namespace mlir::triton
namespace mlir::triton {

template<typename ConcreteOp>
::mlir::TypedValue<mlir::RankedTensorType> detail::DescriptorStoreLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::OpOperand&detail::DescriptorStoreLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrcMutable();
}
template<typename ConcreteOp>
::mlir::TypedValue<mlir::RankedTensorType> detail::DescriptorStoreLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand&detail::DescriptorStoreLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrcMutable(tablegen_opaque_val);
}

} // namespace mlir::triton
namespace mlir::triton {


} // namespace mlir::triton
namespace mlir::triton {

template<typename ConcreteOp>
::mlir::Value detail::PredicatedOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPredicateOperand();
}
template<typename ConcreteOp>
void detail::PredicatedOpInterfaceInterfaceTraits::Model<ConcreteOp>::setPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPredicateOperand(pred);
}
template<typename ConcreteOp>
::mlir::Type detail::PredicatedOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPredicateOperandTypeLike(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPredicateOperandTypeLike();
}
template<typename ConcreteOp>
::mlir::Value detail::PredicatedOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPredicateOperand(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::PredicatedOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setPredicateOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred) {
  return static_cast<const ConcreteOp *>(impl)->setPredicateOperand(tablegen_opaque_val, pred);
}
template<typename ConcreteOp>
::mlir::Type detail::PredicatedOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPredicateOperandTypeLike(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPredicateOperandTypeLike(tablegen_opaque_val);
}

} // namespace mlir::triton
namespace mlir::triton {

template<typename ConcreteOp>
bool detail::EfficientLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getEfficientLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEfficientLayout();
}
template<typename ConcreteOp>
bool detail::EfficientLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getEfficientLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getEfficientLayout(tablegen_opaque_val);
}

} // namespace mlir::triton
namespace mlir::triton {

template<typename ConcreteOp>
::mlir::Value detail::AtomicOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPtr();
}
template<typename ConcreteOp>
::mlir::triton::MemSemantic detail::AtomicOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemSemantic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSem();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPtr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::triton::MemSemantic detail::AtomicOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemSemantic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemSemantic(tablegen_opaque_val);
}

} // namespace mlir::triton
