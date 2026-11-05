/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::gpu {

/// Common trait for all TTGIR layouts.
class LayoutEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait describing shared memory.
class SharedEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
/// It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.
/// 
/// For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
/// For example, for a shape/order pair defines a distribution layout
/// shape = [4, 4]
/// order = [0, 1] // The fastest-changing axis first
/// ->
/// layout = [0  4  8  12]
///          [1  5  9  13]
///          [2  6  10 14]
///          [3  7  11 15]
/// 
/// For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.
/// 
/// If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
/// We call each individual tile "rep".
class DistributedEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait for distributed encodings backed by a LinearLayout
/// (LinearEncodingAttr and GenericLinearEncodingAttr).
class LinearEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

class MmaEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait for all TTGIR layouts.
namespace detail {
struct LayoutEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    CGAEncodingAttr (*getCGALayout)(const Concept *impl, ::mlir::Attribute );
    unsigned (*getRank)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LayoutEncodingTrait;
    Model() : Concept{getCGALayout, getRank} {}

    static inline CGAEncodingAttr getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LayoutEncodingTrait;
    FallbackModel() : Concept{getCGALayout, getRank} {}

    static inline CGAEncodingAttr getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    unsigned getRank(::mlir::Attribute tablegen_opaque_val) const;
  };
};
template <typename ConcreteAttr>
struct LayoutEncodingTraitTrait;

} // namespace detail
class LayoutEncodingTrait : public ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::LayoutEncodingTraitTrait<ConcreteAttr> {};

  /// Get the CGA layout backing this encoding.
  CGAEncodingAttr getCGALayout() const;

  /// Get the rank of the layout.
  unsigned getRank() const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait describing shared memory.
namespace detail {
struct SharedEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    int32_t (*getAlignment)(const Concept *impl, ::mlir::Attribute );
    LinearLayout (*toLinearLayout)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::SharedEncodingTrait;
    Model() : Concept{getAlignment, toLinearLayout} {}

    static inline int32_t getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::SharedEncodingTrait;
    FallbackModel() : Concept{getAlignment, toLinearLayout} {}

    static inline int32_t getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    int32_t getAlignment(::mlir::Attribute tablegen_opaque_val) const;
  };
};
template <typename ConcreteAttr>
struct SharedEncodingTraitTrait;

} // namespace detail
class SharedEncodingTrait : public ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::SharedEncodingTraitTrait<ConcreteAttr> {};

  /// Return the default alignment for the layout.
  int32_t getAlignment() const;

  /// Convert to LinearLayout.
  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
/// It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.
/// 
/// For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
/// For example, for a shape/order pair defines a distribution layout
/// shape = [4, 4]
/// order = [0, 1] // The fastest-changing axis first
/// ->
/// layout = [0  4  8  12]
///          [1  5  9  13]
///          [2  6  10 14]
///          [3  7  11 15]
/// 
/// For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.
/// 
/// If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
/// We call each individual tile "rep".
namespace detail {
struct DistributedEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getRepOrder)(const Concept *impl, ::mlir::Attribute );
    unsigned (*getTotalElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    SmallVector<unsigned> (*getElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    LinearLayout (*toLinearLayout)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::DistributedEncodingTrait;
    Model() : Concept{getRepOrder, getTotalElemsPerThread, getElemsPerThread, toLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::DistributedEncodingTrait;
    FallbackModel() : Concept{getRepOrder, getTotalElemsPerThread, getElemsPerThread, toLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct DistributedEncodingTraitTrait;

} // namespace detail
class DistributedEncodingTrait : public ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DistributedEncodingTraitTrait<ConcreteAttr> {};

  /// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
  SmallVector<unsigned> getRepOrder() const;

  /// Return total element size per thread.
  unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Return element size per thread in each dimension.
  SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Convert to LinearLayout.
  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait for distributed encodings backed by a LinearLayout
/// (LinearEncodingAttr and GenericLinearEncodingAttr).
namespace detail {
struct LinearEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    const LinearLayout &(*getLinearLayout)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*basesPerDim)(const Concept *impl, ::mlir::Attribute , StringAttr, bool);
    SmallVector<unsigned> (*orderPerDim)(const Concept *impl, ::mlir::Attribute , StringAttr, ArrayRef<unsigned>);
    SmallVector<unsigned> (*getOrder)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getContig)(const Concept *impl, ::mlir::Attribute , const char *, SmallVector<unsigned>);
    SmallVector<unsigned> (*getSizePerThread)(const Concept *impl, ::mlir::Attribute );
    CGAEncodingAttr (*getCGALayout)(const Concept *impl, ::mlir::Attribute );
    LinearLayout (*toLinearLayout)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    SmallVector<unsigned> (*getThreadsPerWarp)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getThreadOrder)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getContigPerThread)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getContigPerWarp)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getRepOrder)(const Concept *impl, ::mlir::Attribute );
    SmallVector<unsigned> (*getElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    unsigned (*getTotalElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LinearEncodingTrait;
    Model() : Concept{getLinearLayout, basesPerDim, orderPerDim, getOrder, getContig, getSizePerThread, getCGALayout, toLinearLayout, getThreadsPerWarp, getThreadOrder, getContigPerThread, getContigPerWarp, getRepOrder, getElemsPerThread, getTotalElemsPerThread} {}

    static inline const LinearLayout &getLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> basesPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, bool skipBroadcast);
    static inline SmallVector<unsigned> orderPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, ArrayRef<unsigned> defaultOrder);
    static inline SmallVector<unsigned> getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getContig(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const char * inDim, SmallVector<unsigned> lowerContig);
    static inline SmallVector<unsigned> getSizePerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline CGAEncodingAttr getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getThreadsPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getThreadOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getContigPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getContigPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LinearEncodingTrait;
    FallbackModel() : Concept{getLinearLayout, basesPerDim, orderPerDim, getOrder, getContig, getSizePerThread, getCGALayout, toLinearLayout, getThreadsPerWarp, getThreadOrder, getContigPerThread, getContigPerWarp, getRepOrder, getElemsPerThread, getTotalElemsPerThread} {}

    static inline const LinearLayout &getLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> basesPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, bool skipBroadcast);
    static inline SmallVector<unsigned> orderPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, ArrayRef<unsigned> defaultOrder);
    static inline SmallVector<unsigned> getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getContig(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const char * inDim, SmallVector<unsigned> lowerContig);
    static inline SmallVector<unsigned> getSizePerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline CGAEncodingAttr getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getThreadsPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getThreadOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getContigPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getContigPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    SmallVector<unsigned> basesPerDim(::mlir::Attribute tablegen_opaque_val, StringAttr dimName, bool skipBroadcast) const;
    SmallVector<unsigned> orderPerDim(::mlir::Attribute tablegen_opaque_val, StringAttr dimName, ArrayRef<unsigned> defaultOrder) const;
    SmallVector<unsigned> getOrder(::mlir::Attribute tablegen_opaque_val) const;
    SmallVector<unsigned> getContig(::mlir::Attribute tablegen_opaque_val, const char *inDim, SmallVector<unsigned> lowerContig) const;
    SmallVector<unsigned> getSizePerThread(::mlir::Attribute tablegen_opaque_val) const;
    CGAEncodingAttr getCGALayout(::mlir::Attribute tablegen_opaque_val) const;
    LinearLayout toLinearLayout(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const;
    SmallVector<unsigned> getThreadsPerWarp(::mlir::Attribute tablegen_opaque_val) const;
    SmallVector<unsigned> getThreadOrder(::mlir::Attribute tablegen_opaque_val) const;
    SmallVector<unsigned> getContigPerThread(::mlir::Attribute tablegen_opaque_val) const;
    SmallVector<unsigned> getContigPerWarp(::mlir::Attribute tablegen_opaque_val) const;
    SmallVector<unsigned> getRepOrder(::mlir::Attribute tablegen_opaque_val) const;
    SmallVector<unsigned> getElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const;
    unsigned getTotalElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const;
  };
};
template <typename ConcreteAttr>
struct LinearEncodingTraitTrait;

} // namespace detail
class LinearEncodingTrait : public ::mlir::AttributeInterface<LinearEncodingTrait, detail::LinearEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<LinearEncodingTrait, detail::LinearEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::LinearEncodingTraitTrait<ConcreteAttr> {};

  /// Get the underlying linear layout.
  const LinearLayout &getLinearLayout() const;

  /// Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
  /// Returns the bases of the dimensions `dimName` of the layout.
  /// If skipBroadcast is false, we count a base zero.
  SmallVector<unsigned> basesPerDim(StringAttr dimName, bool skipBroadcast) const;

  /// Generalizes get{Warp,Thread,CTA}Order to linear layouts.
  /// Returns the order of the dimensions `dimName` of the layout.
  /// If more than one dimension is of size one, it uses defaultOrder to determine the order
  /// of the dimensions of size one.
  SmallVector<unsigned> orderPerDim(StringAttr dimName, ArrayRef<unsigned> defaultOrder) const;

  /// Get the layout order.
  SmallVector<unsigned> getOrder() const;

  /// Get contiguity for a given input dimension.
  SmallVector<unsigned> getContig(const char * inDim, SmallVector<unsigned> lowerContig) const;

  /// Get size per thread.
  SmallVector<unsigned> getSizePerThread() const;

  /// Get the CGA layout.
  CGAEncodingAttr getCGALayout() const;

  /// Convert to LinearLayout.
  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  /// Get threads per warp in each dimension.
  SmallVector<unsigned> getThreadsPerWarp() const;

  /// Get the thread order.
  SmallVector<unsigned> getThreadOrder() const;

  /// Get contiguity per thread.
  SmallVector<unsigned> getContigPerThread() const;

  /// Get contiguity per warp.
  SmallVector<unsigned> getContigPerWarp() const;

  /// Get the rep order (also satisfies DistributedEncodingTrait).
  SmallVector<unsigned> getRepOrder() const;

  /// Get elements per thread (overrides DistributedEncodingTrait default).
  SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Get total elements per thread (overrides DistributedEncodingTrait default).
  unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;

    static SmallVector<unsigned> basesPerDim(const LinearLayout &ll,
                                             StringAttr dimName,
                                             bool skipBroadcast);
    static SmallVector<unsigned> orderPerDim(const LinearLayout &ll,
                                             StringAttr dimName,
                                             ArrayRef<unsigned> defaultOrder);
    static SmallVector<unsigned> getContig(const LinearLayout &ll,
                                           const char *inDim,
                                           SmallVector<unsigned> lowerContig,
                                           ArrayRef<unsigned> order);
    static SmallVector<unsigned> getSizePerThread(
        const LinearLayout &ll, ArrayRef<unsigned> cgaSplitNum);
    static CGAEncodingAttr getCGALayout(const LinearLayout &ll);
    static LinearLayout toLinearLayout(const LinearLayout &ll,
                                       ArrayRef<unsigned> repOrder,
                                       ArrayRef<int64_t> shape);
    static SmallVector<unsigned> getElemsPerThread(
        const LinearLayout &ll, ArrayRef<unsigned> repOrder,
        ArrayRef<int64_t> shape);
    static unsigned getTotalElemsPerThread(
        const LinearLayout &ll, ArrayRef<unsigned> repOrder,
        ArrayRef<int64_t> shape);
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

namespace detail {
struct MmaEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getRepOrderForOperand)(const Concept *impl, ::mlir::Attribute , int);
    LinearLayout (*dotOperandToLinearLayout)(const Concept *impl, ::mlir::Attribute , ::mlir::Attribute, ArrayRef<int64_t>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MmaEncodingTrait;
    Model() : Concept{getRepOrderForOperand, dotOperandToLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx);
    static inline LinearLayout dotOperandToLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Attribute dotOp, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MmaEncodingTrait;
    FallbackModel() : Concept{getRepOrderForOperand, dotOperandToLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx);
    static inline LinearLayout dotOperandToLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Attribute dotOp, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    LinearLayout dotOperandToLinearLayout(::mlir::Attribute tablegen_opaque_val, ::mlir::Attribute dotOp, ArrayRef<int64_t> shape) const;
  };
};
template <typename ConcreteAttr>
struct MmaEncodingTraitTrait;

} // namespace detail
class MmaEncodingTrait : public ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::MmaEncodingTraitTrait<ConcreteAttr> {};

  /// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
  SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;

  /// Convert a dot operand whose parent is this MMA layout to a LinearLayout.
  LinearLayout dotOperandToLinearLayout(::mlir::Attribute dotOp, ArrayRef<int64_t> shape) const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct LayoutEncodingTraitTrait : public ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Get the rank of the layout.
    unsigned getRank() const {
      return (*static_cast<const ConcreteAttr *>(this)).getCGALayout().getRank();
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct SharedEncodingTraitTrait : public ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Return the default alignment for the layout.
    int32_t getAlignment() const {
      return 16;
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct DistributedEncodingTraitTrait : public ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct LinearEncodingTraitTrait : public ::mlir::AttributeInterface<LinearEncodingTrait, detail::LinearEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
    /// Returns the bases of the dimensions `dimName` of the layout.
    /// If skipBroadcast is false, we count a base zero.
    SmallVector<unsigned> basesPerDim(StringAttr dimName, bool skipBroadcast) const {
      return LinearEncodingTrait::basesPerDim(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), dimName, skipBroadcast);
    }

    /// Generalizes get{Warp,Thread,CTA}Order to linear layouts.
    /// Returns the order of the dimensions `dimName` of the layout.
    /// If more than one dimension is of size one, it uses defaultOrder to determine the order
    /// of the dimensions of size one.
    SmallVector<unsigned> orderPerDim(StringAttr dimName, ArrayRef<unsigned> defaultOrder) const {
      return LinearEncodingTrait::orderPerDim(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), dimName, defaultOrder);
    }

    /// Get the layout order.
    SmallVector<unsigned> getOrder() const {
      const auto &ll = (*static_cast<const ConcreteAttr *>(this)).getLinearLayout();
      auto rank = ll.getNumOutDims();
      SmallVector<unsigned> order(rank);
      // Choose [rank-1, rank-2, ... 0] as the default order in case
      // there are dims that do not move in the register.
      // This order is as good as any.
      std::iota(order.rbegin(), order.rend(), 0);
      return (*static_cast<const ConcreteAttr *>(this)).orderPerDim(
          StringAttr::get((*static_cast<const ConcreteAttr *>(this)).getContext(), "register"), order);
    }

    /// Get contiguity for a given input dimension.
    SmallVector<unsigned> getContig(const char * inDim, SmallVector<unsigned> lowerContig) const {
      return LinearEncodingTrait::getContig(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), inDim, lowerContig, (*static_cast<const ConcreteAttr *>(this)).getOrder());
    }

    /// Get size per thread.
    SmallVector<unsigned> getSizePerThread() const {
      auto cgaSplitNum = (*static_cast<const ConcreteAttr *>(this)).getCGALayout().getCTASplitNum();
      return LinearEncodingTrait::getSizePerThread(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), cgaSplitNum);
    }

    /// Get the CGA layout.
    CGAEncodingAttr getCGALayout() const {
      return LinearEncodingTrait::getCGALayout((*static_cast<const ConcreteAttr *>(this)).getLinearLayout());
    }

    /// Convert to LinearLayout.
    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const {
      auto repOrder = (*static_cast<const ConcreteAttr *>(this)).getRepOrder();
      return LinearEncodingTrait::toLinearLayout(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), repOrder, shape);
    }

    /// Get threads per warp in each dimension.
    SmallVector<unsigned> getThreadsPerWarp() const {
      return (*static_cast<const ConcreteAttr *>(this)).basesPerDim(
          StringAttr::get((*static_cast<const ConcreteAttr *>(this)).getContext(), "lane"), /*skipBroadcast=*/true);
    }

    /// Get the thread order.
    SmallVector<unsigned> getThreadOrder() const {
      return (*static_cast<const ConcreteAttr *>(this)).orderPerDim(
          StringAttr::get((*static_cast<const ConcreteAttr *>(this)).getContext(), "lane"), (*static_cast<const ConcreteAttr *>(this)).getOrder());
    }

    /// Get contiguity per thread.
    SmallVector<unsigned> getContigPerThread() const {
      SmallVector<unsigned> c((*static_cast<const ConcreteAttr *>(this)).getOrder().size(), 1);
      return (*static_cast<const ConcreteAttr *>(this)).getContig("register", c);
    }

    /// Get contiguity per warp.
    SmallVector<unsigned> getContigPerWarp() const {
      return (*static_cast<const ConcreteAttr *>(this)).getContig("lane", (*static_cast<const ConcreteAttr *>(this)).getContigPerThread());
    }

    /// Get the rep order (also satisfies DistributedEncodingTrait).
    SmallVector<unsigned> getRepOrder() const {
      // This is not correct, but:
      // - It happens to agree in most places with the legacy layout
      // - getRepOrder does not make sense for LinearEncodingAttr as it already has
      //   the same shape as the tensor that uses it
      return (*static_cast<const ConcreteAttr *>(this)).getOrder();
    }

    /// Get elements per thread (overrides DistributedEncodingTrait default).
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const {
      return LinearEncodingTrait::getElemsPerThread(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), (*static_cast<const ConcreteAttr *>(this)).getOrder(), shape);
    }

    /// Get total elements per thread (overrides DistributedEncodingTrait default).
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
      return LinearEncodingTrait::getTotalElemsPerThread(
          (*static_cast<const ConcreteAttr *>(this)).getLinearLayout(), (*static_cast<const ConcreteAttr *>(this)).getOrder(), shape);
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct MmaEncodingTraitTrait : public ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Convert a dot operand whose parent is this MMA layout to a LinearLayout.
    LinearLayout dotOperandToLinearLayout(::mlir::Attribute dotOp, ArrayRef<int64_t> shape) const {
      llvm::report_fatal_error(
                          "this MMA layout is not a valid dot-operand parent");
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
CGAEncodingAttr detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCGALayout();
}
template<typename ConcreteAttr>
unsigned detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRank();
}
template<typename ConcreteAttr>
CGAEncodingAttr detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getCGALayout(tablegen_opaque_val);
}
template<typename ConcreteAttr>
unsigned detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRank(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteAttr>
unsigned detail::LayoutEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getRank(::mlir::Attribute tablegen_opaque_val) const {
return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCGALayout().getRank();
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getAlignment();
}
template<typename ConcreteAttr>
LinearLayout detail::SharedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).toLinearLayout(shape);
}
template<typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getAlignment(tablegen_opaque_val);
}
template<typename ConcreteAttr>
LinearLayout detail::SharedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->toLinearLayout(tablegen_opaque_val, shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getAlignment(::mlir::Attribute tablegen_opaque_val) const {
return 16;
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrder();
}
template<typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return toLinearEncoding((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), shape).getTotalElemsPerThread(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return toLinearEncoding((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), shape).getElemsPerThread(shape);
}
template<typename ConcreteAttr>
LinearLayout detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).toLinearLayout(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getTotalElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
LinearLayout detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->toLinearLayout(tablegen_opaque_val, shape);
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
const LinearLayout &detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::basesPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, bool skipBroadcast) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).basesPerDim(dimName, skipBroadcast);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::orderPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, ArrayRef<unsigned> defaultOrder) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).orderPerDim(dimName, defaultOrder);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getContig(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const char * inDim, SmallVector<unsigned> lowerContig) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContig(inDim, lowerContig);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getSizePerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getSizePerThread();
}
template<typename ConcreteAttr>
CGAEncodingAttr detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCGALayout();
}
template<typename ConcreteAttr>
LinearLayout detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).toLinearLayout(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getThreadsPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getThreadsPerWarp();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getThreadOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getThreadOrder();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getContigPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContigPerThread();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getContigPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContigPerWarp();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrder();
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getElemsPerThread(shape);
}
template<typename ConcreteAttr>
unsigned detail::LinearEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getTotalElemsPerThread(shape);
}
template<typename ConcreteAttr>
const LinearLayout &detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getLinearLayout(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::basesPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, bool skipBroadcast) {
  return static_cast<const ConcreteAttr *>(impl)->basesPerDim(tablegen_opaque_val, dimName, skipBroadcast);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::orderPerDim(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, StringAttr dimName, ArrayRef<unsigned> defaultOrder) {
  return static_cast<const ConcreteAttr *>(impl)->orderPerDim(tablegen_opaque_val, dimName, defaultOrder);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getContig(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const char * inDim, SmallVector<unsigned> lowerContig) {
  return static_cast<const ConcreteAttr *>(impl)->getContig(tablegen_opaque_val, inDim, lowerContig);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getSizePerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getSizePerThread(tablegen_opaque_val);
}
template<typename ConcreteAttr>
CGAEncodingAttr detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getCGALayout(tablegen_opaque_val);
}
template<typename ConcreteAttr>
LinearLayout detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->toLinearLayout(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getThreadsPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getThreadsPerWarp(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getThreadOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getThreadOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getContigPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getContigPerThread(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getContigPerWarp(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getContigPerWarp(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
unsigned detail::LinearEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getTotalElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::basesPerDim(::mlir::Attribute tablegen_opaque_val, StringAttr dimName, bool skipBroadcast) const {
return LinearEncodingTrait::basesPerDim(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), dimName, skipBroadcast);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::orderPerDim(::mlir::Attribute tablegen_opaque_val, StringAttr dimName, ArrayRef<unsigned> defaultOrder) const {
return LinearEncodingTrait::orderPerDim(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), dimName, defaultOrder);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getOrder(::mlir::Attribute tablegen_opaque_val) const {
const auto &ll = (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout();
      auto rank = ll.getNumOutDims();
      SmallVector<unsigned> order(rank);
      // Choose [rank-1, rank-2, ... 0] as the default order in case
      // there are dims that do not move in the register.
      // This order is as good as any.
      std::iota(order.rbegin(), order.rend(), 0);
      return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).orderPerDim(
          StringAttr::get((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContext(), "register"), order);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getContig(::mlir::Attribute tablegen_opaque_val, const char *inDim, SmallVector<unsigned> lowerContig) const {
return LinearEncodingTrait::getContig(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), inDim, lowerContig, (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder());
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getSizePerThread(::mlir::Attribute tablegen_opaque_val) const {
auto cgaSplitNum = (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCGALayout().getCTASplitNum();
      return LinearEncodingTrait::getSizePerThread(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), cgaSplitNum);
}
template<typename ConcreteModel, typename ConcreteAttr>
CGAEncodingAttr detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getCGALayout(::mlir::Attribute tablegen_opaque_val) const {
return LinearEncodingTrait::getCGALayout((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout());
}
template<typename ConcreteModel, typename ConcreteAttr>
LinearLayout detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::toLinearLayout(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const {
auto repOrder = (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrder();
      return LinearEncodingTrait::toLinearLayout(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), repOrder, shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getThreadsPerWarp(::mlir::Attribute tablegen_opaque_val) const {
return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).basesPerDim(
          StringAttr::get((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContext(), "lane"), /*skipBroadcast=*/true);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getThreadOrder(::mlir::Attribute tablegen_opaque_val) const {
return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).orderPerDim(
          StringAttr::get((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContext(), "lane"), (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder());
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getContigPerThread(::mlir::Attribute tablegen_opaque_val) const {
SmallVector<unsigned> c((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder().size(), 1);
      return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContig("register", c);
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getContigPerWarp(::mlir::Attribute tablegen_opaque_val) const {
return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContig("lane", (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getContigPerThread());
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getRepOrder(::mlir::Attribute tablegen_opaque_val) const {
// This is not correct, but:
      // - It happens to agree in most places with the legacy layout
      // - getRepOrder does not make sense for LinearEncodingAttr as it already has
      //   the same shape as the tensor that uses it
      return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder();
}
template<typename ConcreteModel, typename ConcreteAttr>
SmallVector<unsigned> detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const {
return LinearEncodingTrait::getElemsPerThread(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder(), shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
unsigned detail::LinearEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getTotalElemsPerThread(::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) const {
return LinearEncodingTrait::getTotalElemsPerThread(
          (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getLinearLayout(), (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder(), shape);
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
SmallVector<unsigned> detail::MmaEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrderForOperand(opIdx);
}
template<typename ConcreteAttr>
LinearLayout detail::MmaEncodingTraitInterfaceTraits::Model<ConcreteAttr>::dotOperandToLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Attribute dotOp, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).dotOperandToLinearLayout(dotOp, shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::MmaEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrderForOperand(tablegen_opaque_val, opIdx);
}
template<typename ConcreteAttr>
LinearLayout detail::MmaEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::dotOperandToLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Attribute dotOp, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->dotOperandToLinearLayout(tablegen_opaque_val, dotOp, shape);
}
template<typename ConcreteModel, typename ConcreteAttr>
LinearLayout detail::MmaEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::dotOperandToLinearLayout(::mlir::Attribute tablegen_opaque_val, ::mlir::Attribute dotOp, ArrayRef<int64_t> shape) const {
llvm::report_fatal_error(
                          "this MMA layout is not a valid dot-operand parent");
}

} // namespace mlir::triton::gpu
