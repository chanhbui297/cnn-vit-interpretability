/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvidia_gpu {

/// Common interface for asynchronous TMA operations.
class TMAOpInterface;

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// Common interface for asynchronous TMA operations that write shared memory.
class TMALoadLikeOpInterface;

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// Common interface for asynchronous TMA operations that read shared memory.
class TMAStoreLikeOpInterface;

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// This interface is implemented by MMAv5 dot and dot scaled ops.
class MMAv5OpInterface;

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// Common interface for asynchronous TMA operations.
namespace detail {
struct TMAOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getDesc)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::TMAOpInterface;
    Model() : Concept{getDesc} {}

    static inline ::mlir::Value getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::TMAOpInterface;
    FallbackModel() : Concept{getDesc} {}

    static inline ::mlir::Value getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct TMAOpInterfaceTrait;

} // namespace detail
class TMAOpInterface : public ::mlir::OpInterface<TMAOpInterface, detail::TMAOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TMAOpInterface, detail::TMAOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TMAOpInterfaceTrait<ConcreteOp> {};

  /// Get the tensor descriptor
  ::mlir::Value getDesc();
};

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// Common interface for asynchronous TMA operations that write shared memory.
namespace detail {
struct TMALoadLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getResult)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getBarrier)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getPred)(const Concept *impl, ::mlir::Operation *);
    bool (*getMulticast)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::triton::nvidia_gpu::TMAOpInterface::Concept *implTMAOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implTMAOpInterface = interfaceMap.lookup<::mlir::triton::nvidia_gpu::TMAOpInterface>();
      assert(implTMAOpInterface && "`::mlir::triton::nvidia_gpu::TMALoadLikeOpInterface` expected its base interface `::mlir::triton::nvidia_gpu::TMAOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::TMALoadLikeOpInterface;
    Model() : Concept{getResult, getBarrier, getPred, getMulticast} {}

    static inline ::mlir::Value getResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getPred(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::TMALoadLikeOpInterface;
    FallbackModel() : Concept{getResult, getBarrier, getPred, getMulticast} {}

    static inline ::mlir::Value getResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getPred(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct TMALoadLikeOpInterfaceTrait;

} // namespace detail
class TMALoadLikeOpInterface : public ::mlir::OpInterface<TMALoadLikeOpInterface, detail::TMALoadLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TMALoadLikeOpInterface, detail::TMALoadLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TMALoadLikeOpInterfaceTrait<ConcreteOp> {};

  /// Get the destination memory descriptor
  ::mlir::Value getResult();

  /// Get the completion barrier
  ::mlir::Value getBarrier();

  /// Get the predicate
  ::mlir::Value getPred();

  /// Return true if this load uses multicast
  bool getMulticast();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::triton::nvidia_gpu::TMAOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::triton::nvidia_gpu::TMAOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::triton::nvidia_gpu::TMAOpInterface(*this, getImpl()->implTMAOpInterface);
  }


  /// Get the tensor descriptor
  ::mlir::Value getDesc();
};

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// Common interface for asynchronous TMA operations that read shared memory.
namespace detail {
struct TMAStoreLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand&(*getSrcMutable)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::triton::nvidia_gpu::TMAOpInterface::Concept *implTMAOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implTMAOpInterface = interfaceMap.lookup<::mlir::triton::nvidia_gpu::TMAOpInterface>();
      assert(implTMAOpInterface && "`::mlir::triton::nvidia_gpu::TMAStoreLikeOpInterface` expected its base interface `::mlir::triton::nvidia_gpu::TMAOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::TMAStoreLikeOpInterface;
    Model() : Concept{getSrc, getSrcMutable} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand&getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::TMAStoreLikeOpInterface;
    FallbackModel() : Concept{getSrc, getSrcMutable} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand&getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct TMAStoreLikeOpInterfaceTrait;

} // namespace detail
class TMAStoreLikeOpInterface : public ::mlir::OpInterface<TMAStoreLikeOpInterface, detail::TMAStoreLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TMAStoreLikeOpInterface, detail::TMAStoreLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TMAStoreLikeOpInterfaceTrait<ConcreteOp> {};

  /// Get the source memory descriptor
  ::mlir::Value getSrc();

  /// Get mutable source memory descriptor
  ::mlir::OpOperand&getSrcMutable();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::triton::nvidia_gpu::TMAOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::triton::nvidia_gpu::TMAOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::triton::nvidia_gpu::TMAOpInterface(*this, getImpl()->implTMAOpInterface);
  }


  /// Get the tensor descriptor
  ::mlir::Value getDesc();
};

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// This interface is implemented by MMAv5 dot and dot scaled ops.
namespace detail {
struct MMAv5OpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getA)(const Concept *impl, ::mlir::Operation *);
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getB)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*useAccumulator)(const Concept *impl, ::mlir::Operation *);
    void (*setUseAccumulator)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::ValueRange (*getCompletionBarriers)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ValueRange (*getCompletionBarrierPreds)(const Concept *impl, ::mlir::Operation *);
    ::llvm::SmallVector<::mlir::Value> (*getCompletionDescs)(const Concept *impl, ::mlir::Operation *);
    void (*addCompletionBarrier)(const Concept *impl, ::mlir::Operation *, ::mlir::Value, ::mlir::Value);
    bool (*getMulticast)(const Concept *impl, ::mlir::Operation *);
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getAccumulator)(const Concept *impl, ::mlir::Operation *);
    void (*setAccumulator)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::Value (*getPredicate)(const Concept *impl, ::mlir::Operation *);
    void (*setPredicate)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::Value (*getAccDep)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MutableOperandRange (*getAccDepMutable)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getToken)(const Concept *impl, ::mlir::Operation *);
    void (*setIsAsync)(const Concept *impl, ::mlir::Operation *, bool);
    bool (*getTwoCtas)(const Concept *impl, ::mlir::Operation *);
    bool (*isAsync)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::MMAv5OpInterface;
    Model() : Concept{getA, getB, useAccumulator, setUseAccumulator, getCompletionBarriers, getCompletionBarrierPreds, getCompletionDescs, addCompletionBarrier, getMulticast, getAccumulator, setAccumulator, getPredicate, setPredicate, getAccDep, getAccDepMutable, getToken, setIsAsync, getTwoCtas, isAsync} {}

    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag);
    static inline ::mlir::ValueRange getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ValueRange getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::Value> getCompletionDescs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred);
    static inline bool getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum);
    static inline ::mlir::Value getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred);
    static inline ::mlir::Value getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync);
    static inline bool getTwoCtas(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::MMAv5OpInterface;
    FallbackModel() : Concept{getA, getB, useAccumulator, setUseAccumulator, getCompletionBarriers, getCompletionBarrierPreds, getCompletionDescs, addCompletionBarrier, getMulticast, getAccumulator, setAccumulator, getPredicate, setPredicate, getAccDep, getAccDepMutable, getToken, setIsAsync, getTwoCtas, isAsync} {}

    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag);
    static inline ::mlir::ValueRange getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ValueRange getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::Value> getCompletionDescs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred);
    static inline bool getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum);
    static inline ::mlir::Value getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred);
    static inline ::mlir::Value getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync);
    static inline bool getTwoCtas(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct MMAv5OpInterfaceTrait;

} // namespace detail
class MMAv5OpInterface : public ::mlir::OpInterface<MMAv5OpInterface, detail::MMAv5OpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MMAv5OpInterface, detail::MMAv5OpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MMAv5OpInterfaceTrait<ConcreteOp> {};

  /// Return the A operand.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getA();

  /// Return the B operand.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getB();

  /// Return the accumulator init flag.
  ::mlir::Value useAccumulator();

  /// Set the accumulator init flag.
  void setUseAccumulator(::mlir::Value flag);

  /// Return the completion barriers of this MMAv5 op.
  ::mlir::ValueRange getCompletionBarriers();

  /// Return the completion barrier predicates of this MMAv5 op.
  ::mlir::ValueRange getCompletionBarrierPreds();

  /// Return the completion descriptor operands of this MMAv5 op.
  ::llvm::SmallVector<::mlir::Value> getCompletionDescs();

  /// Associate a new completion barrier to this MMAv5 op.
  void addCompletionBarrier(::mlir::Value barrier, ::mlir::Value pred);

  /// Return true if this MMAv5 op uses multicast completion metadata.
  bool getMulticast();

  /// Return the accumulator.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getAccumulator();

  /// Set the accumulator.
  void setAccumulator(::mlir::Value accum);

  /// Return the predicate of this op.
  ::mlir::Value getPredicate();

  /// Set the predicate of this op.
  void setPredicate(::mlir::Value pred);

  /// Get the memory dependencies of the accumulator.
  ::mlir::Value getAccDep();

  /// Get the mutable memory dependencies of the accumulator.
  ::mlir::MutableOperandRange getAccDepMutable();

  /// Get the produced write dependency of the accumulator.
  ::mlir::Value getToken();

  /// Indicate that this MMA op executes asynchronously.
  void setIsAsync(bool isAsync);

  /// Return true if this MMA op uses two CTAs.
  bool getTwoCtas();

  /// Return true if this MMA op executes asynchronously.
  bool isAsync();
};

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu::detail {

  template <typename ConcreteOp>
  struct TMAOpInterfaceTrait : public ::mlir::OpInterface<TMAOpInterface, detail::TMAOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::nvidia_gpu::detail
namespace mlir::triton::nvidia_gpu::detail {

  template <typename ConcreteOp>
  struct TMALoadLikeOpInterfaceTrait : public ::mlir::OpInterface<TMALoadLikeOpInterface, detail::TMALoadLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::nvidia_gpu::detail
namespace mlir::triton::nvidia_gpu::detail {

  template <typename ConcreteOp>
  struct TMAStoreLikeOpInterfaceTrait : public ::mlir::OpInterface<TMAStoreLikeOpInterface, detail::TMAStoreLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::nvidia_gpu::detail
namespace mlir::triton::nvidia_gpu::detail {

  template <typename ConcreteOp>
  struct MMAv5OpInterfaceTrait : public ::mlir::OpInterface<MMAv5OpInterface, detail::MMAv5OpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::triton::nvidia_gpu::impl::verifyMMAv5Op(op);
    }
  };

} // namespace mlir::triton::nvidia_gpu::detail
namespace mlir::triton::nvidia_gpu {

template<typename ConcreteOp>
::mlir::Value detail::TMAOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDesc();
}
template<typename ConcreteOp>
::mlir::Value detail::TMAOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDesc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDesc(tablegen_opaque_val);
}

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

template<typename ConcreteOp>
::mlir::Value detail::TMALoadLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResult();
}
template<typename ConcreteOp>
::mlir::Value detail::TMALoadLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBarrier();
}
template<typename ConcreteOp>
::mlir::Value detail::TMALoadLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPred(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPred();
}
template<typename ConcreteOp>
bool detail::TMALoadLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMulticast();
}
template<typename ConcreteOp>
::mlir::Value detail::TMALoadLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResult(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::TMALoadLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBarrier(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::TMALoadLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPred(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPred(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::TMALoadLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMulticast(tablegen_opaque_val);
}

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

template<typename ConcreteOp>
::mlir::Value detail::TMAStoreLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::OpOperand&detail::TMAStoreLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrcMutable();
}
template<typename ConcreteOp>
::mlir::Value detail::TMAStoreLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand&detail::TMAStoreLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSrcMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrcMutable(tablegen_opaque_val);
}

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getA();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getB();
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).useAccumulator();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setUseAccumulator(flag);
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCompletionBarriers();
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCompletionBarrierPreds();
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getCompletionDescs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCompletionDescs();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addCompletionBarrier(barrier, pred);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMulticast();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccumulator();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setAccumulator(accum);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPredicate();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPredicate(pred);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccDep();
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccDepMutable();
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getToken();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setIsAsync(isAsync);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getTwoCtas(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTwoCtas();
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isAsync();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getA(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getB(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->useAccumulator(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag) {
  return static_cast<const ConcreteOp *>(impl)->setUseAccumulator(tablegen_opaque_val, flag);
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCompletionBarriers(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCompletionBarrierPreds(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCompletionDescs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCompletionDescs(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred) {
  return static_cast<const ConcreteOp *>(impl)->addCompletionBarrier(tablegen_opaque_val, barrier, pred);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMulticast(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMulticast(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccumulator(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum) {
  return static_cast<const ConcreteOp *>(impl)->setAccumulator(tablegen_opaque_val, accum);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPredicate(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred) {
  return static_cast<const ConcreteOp *>(impl)->setPredicate(tablegen_opaque_val, pred);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccDep(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccDepMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getToken(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync) {
  return static_cast<const ConcreteOp *>(impl)->setIsAsync(tablegen_opaque_val, isAsync);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTwoCtas(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTwoCtas(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isAsync(tablegen_opaque_val);
}

} // namespace mlir::triton::nvidia_gpu
