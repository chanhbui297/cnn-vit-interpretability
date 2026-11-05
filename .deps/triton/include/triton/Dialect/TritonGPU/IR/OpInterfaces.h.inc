/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::gpu {

/// This interface is for operations that cast floating-point numbers,
/// either upcasting or downcasting.
class CastFpOpInterface;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Interface for operations that use mbarrier synchronization primitives.
/// Returns the barrier memdesc operand (or null if absent), allowing
/// backend-agnostic barrier identification without direct dialect dependencies.
class MBarrierOpInterface;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// This interface is for operations that cast floating-point numbers,
/// either upcasting or downcasting.
namespace detail {
struct CastFpOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::Attribute (*inferDstEncoding)(const Concept *impl, ::mlir::Operation *, unsigned, mlir::Attribute);
    mlir::Attribute (*inferSrcEncoding)(const Concept *impl, ::mlir::Operation *, unsigned, mlir::Attribute);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::CastFpOpInterface;
    Model() : Concept{inferDstEncoding, inferSrcEncoding} {}

    static inline mlir::Attribute inferDstEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute srcEnc);
    static inline mlir::Attribute inferSrcEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute dstEnc);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::CastFpOpInterface;
    FallbackModel() : Concept{inferDstEncoding, inferSrcEncoding} {}

    static inline mlir::Attribute inferDstEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute srcEnc);
    static inline mlir::Attribute inferSrcEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute dstEnc);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct CastFpOpInterfaceTrait;

} // namespace detail
class CastFpOpInterface : public ::mlir::OpInterface<CastFpOpInterface, detail::CastFpOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CastFpOpInterface, detail::CastFpOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CastFpOpInterfaceTrait<ConcreteOp> {};

  /// Infer destination encoding
  mlir::Attribute inferDstEncoding(unsigned opIdx, mlir::Attribute srcEnc);

  /// Infer operand encoding from dst encoding
  mlir::Attribute inferSrcEncoding(unsigned opIdx, mlir::Attribute dstEnc);
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Interface for operations that use mbarrier synchronization primitives.
/// Returns the barrier memdesc operand (or null if absent), allowing
/// backend-agnostic barrier identification without direct dialect dependencies.
namespace detail {
struct MBarrierOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getBarrier)(const Concept *impl, ::mlir::Operation *);
    ::llvm::SmallVector<::mlir::Value> (*getBarriers)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MBarrierOpInterface;
    Model() : Concept{getBarrier, getBarriers} {}

    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::Value> getBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MBarrierOpInterface;
    FallbackModel() : Concept{getBarrier, getBarriers} {}

    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::Value> getBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::SmallVector<::mlir::Value> getBarriers(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct MBarrierOpInterfaceTrait;

} // namespace detail
class MBarrierOpInterface : public ::mlir::OpInterface<MBarrierOpInterface, detail::MBarrierOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MBarrierOpInterface, detail::MBarrierOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MBarrierOpInterfaceTrait<ConcreteOp> {};

  /// Return the barrier memdesc operand, or null if absent.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier();

  /// Return all barrier memdesc operands.
  ::llvm::SmallVector<::mlir::Value> getBarriers();
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu::detail {

  template <typename ConcreteOp>
  struct CastFpOpInterfaceTrait : public ::mlir::OpInterface<CastFpOpInterface, detail::CastFpOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteOp>
  struct MBarrierOpInterfaceTrait : public ::mlir::OpInterface<MBarrierOpInterface, detail::MBarrierOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {

    /// Return the barrier memdesc operand, or null if absent.
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier() {
      return {};
    }

    /// Return all barrier memdesc operands.
    ::llvm::SmallVector<::mlir::Value> getBarriers() {
      if (::mlir::Value barrier = (*static_cast<ConcreteOp *>(this)).getBarrier())
          return {barrier};
        return {};
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::triton::gpu::verifyMBarrierOpInterface(op);
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu {

template<typename ConcreteOp>
mlir::Attribute detail::CastFpOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferDstEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute srcEnc) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferDstEncoding(opIdx, srcEnc);
}
template<typename ConcreteOp>
mlir::Attribute detail::CastFpOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferSrcEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute dstEnc) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferSrcEncoding(opIdx, dstEnc);
}
template<typename ConcreteOp>
mlir::Attribute detail::CastFpOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferDstEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute srcEnc) {
  return static_cast<const ConcreteOp *>(impl)->inferDstEncoding(tablegen_opaque_val, opIdx, srcEnc);
}
template<typename ConcreteOp>
mlir::Attribute detail::CastFpOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferSrcEncoding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned opIdx, mlir::Attribute dstEnc) {
  return static_cast<const ConcreteOp *>(impl)->inferSrcEncoding(tablegen_opaque_val, opIdx, dstEnc);
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MBarrierOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBarrier();
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::MBarrierOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBarriers();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MBarrierOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBarrier(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::MBarrierOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBarriers(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MBarrierOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getBarrier(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::MBarrierOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getBarriers(::mlir::Operation *tablegen_opaque_val) const {
if (::mlir::Value barrier = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBarrier())
          return {barrier};
        return {};
}

} // namespace mlir::triton::gpu
