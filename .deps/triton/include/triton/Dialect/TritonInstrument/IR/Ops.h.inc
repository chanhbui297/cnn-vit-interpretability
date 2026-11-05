/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentOps.td                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::instrument {

/// non-saturating NVIDIA MMAv2 i8 dot
/// Performs a wrapping i8 matrix multiplication into an i32 accumulator using
/// NVIDIA MMAv2. The A and B operands have independent signedness.
class DotI8Op;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// assert the uniform condition
/// Assert that the condition is true given all threads in the warp group have
/// the same value, so only one thread needs to evaluate the assert and print
/// the message.
class ExperimentalAssertUniformOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// define an array of buffer descriptors
/// Create a tensor of buffer descriptors packing 32-bit pointer offsets and
/// 32-bit lengths into 64-bit elements.
class ExperimentalBufferDescriptorsOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Get the CTA id within the current cluster
/// Return the cluster-local CTA id used to index ConSan's multi-CTA scratch
/// slabs. For single-CTA kernels this is always zero.
class ExperimentalClusterCTAIdOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Embed float value in the fpsan integer ring
class ExperimentalFPSanEmbedOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Unembed fpsan integer payload as a float value
class ExperimentalFPSanUnembedOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Lower a GSan-instrumented atomic cas
class ExperimentalGSanAtomicCASOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Lower a GSan-instrumented atomic load
class ExperimentalGSanAtomicLoadOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Instrument a completed atomic poll for GSan
/// Records the successful read performed by an atomic poll after the hardware
/// polling loop has completed.
class ExperimentalGSanAtomicPollOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Lower a GSan-instrumented atomic rmw
class ExperimentalGSanAtomicRMWOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Lower a GSan-instrumented atomic store
class ExperimentalGSanAtomicStoreOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Instrument an atomic TMA descriptor access for GSan
/// Passes a native tensor descriptor and coordinates to the GSan runtime for
/// atomic instrumentation with the specified memory semantics and scope.
class ExperimentalGSanAtomicTensorDescAccessOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Initialize a GSan cluster-barrier rendezvous
/// Initializes cluster-shared profile scratch used to synchronize GSan vector
/// clocks. This operation is internal to GSan instrumentation.
class ExperimentalGSanClusterBarrierInitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Synchronize GSan clocks at a cluster barrier
/// Joins the vector clocks of all CTAs participating in a non-relaxed cluster
/// barrier. This operation is internal to GSan instrumentation.
class ExperimentalGSanClusterBarrierSyncOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Acquire completed predecessor clocks after a grid dependency wait
class ExperimentalGSanGridDependencyWaitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Instrument an indexed TMA gather or scatter for GSan
/// Passes a native tensor descriptor and its row indices directly to the GSan
/// runtime, which checks row and column bounds before updating shadow memory.
class ExperimentalGSanIndexedTensorDescAccessOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Initialize GSan thread
class ExperimentalGSanInitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Publish a completed GSan thread clock to its launch
class ExperimentalGSanKernelExitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Record an mbarrier arrival in GSan
/// Records an arrival for every mbarrier targeted by the corresponding
/// hardware operation. When `publishClock` is true, the issuing CTA's vector
/// clock is also published without advancing its local epoch. Waiters acquire
/// the issuing CTA's completed epochs and its imported clocks, but not its
/// current epoch. A false value records only phase
/// progress for asynchronous completion signals that provide no generic
/// memory ordering. `multicastMasks` contains CTA-id broadcast masks whose
/// recipient bitsets are unioned. When `multicast` is false, the arrival
/// targets the leader selected by the barrier's CGA layout.
/// This operation is internal to GSan instrumentation.
class ExperimentalGSanMBarrierArriveOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Initialize an mbarrier in GSan
/// Associates an mbarrier with its expected arrival count.
/// This operation is internal to GSan instrumentation.
class ExperimentalGSanMBarrierInitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Initialize GSan mbarrier release state
/// Initializes GSan's internal mbarrier table, used to sequence memory accesses
/// through cluster mbarriers. This operation is internal to GSan instrumentation.
class ExperimentalGSanMBarrierTableInitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Acquire GSan clocks after an mbarrier wait
/// Acquires the clocks published to the completed mbarrier phase, excluding
/// each publisher's current epoch. Global-memory handoffs that rely solely on
/// an mbarrier may therefore be conservatively reported as races.
/// This operation is internal to GSan instrumentation.
class ExperimentalGSanMBarrierWaitOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Instrument a tensor load/store access for GSan
/// Emits runtime instrumentation for a tensor pointer access. The pointer and
/// optional mask are consumed by the GSan runtime.
class ExperimentalGSanTensorAccessOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Instrument a TMA tensor descriptor access for GSan
/// Passes the native tensor descriptor and its coordinates directly to the
/// GSan runtime, which decodes the descriptor and iterates over the tile.
class ExperimentalGSanTensorDescAccessOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Gather elements from shared memory with logical base offsets
/// Gather elements from a shared memory descriptor using an index tensor along
/// one axis, after shifting the logical source coordinates by rank-sized scalar
/// offsets. This is intentionally private to instrumentation passes.
class ExperimentalLocalGatherOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Acquire a lock.
/// Enter a critical section by acquiring a lock with single thread.
class ExperimentalLockAcquireOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Release a lock.
/// Leave a critical section by releasing a lock with single thread.
class ExperimentalLockReleaseOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Convert a memdesc into its base pointer as i32
/// Extract the base pointer from the given memdesc and return it as a 32-bit
/// integer. This can be used to compare the memdesc against tensors of barrier
/// pointers maintained by the concurrency sanitizer.
class ExperimentalMemDescToI32Op;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Materialize a memory-space offset as an i32 address
/// Add a static byte or encoded tensor-memory offset to the current
/// function's memory-space base and return the 32-bit address used by ConSan.
/// This provides the runtime identity corresponding to a compile-time
/// BufferRegion key without materializing a descriptor table.
class ExperimentalMemoryOffsetToI32Op;

} // namespace mlir::triton::instrument
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::DotI8Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class DotI8OpGenericAdaptorBase {
public:
  struct Properties {
    using aSignedTy = ::mlir::BoolAttr;
    aSignedTy aSigned;

    auto getASigned() const {
      auto &propStorage = this->aSigned;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setASigned(const ::mlir::BoolAttr &propValue) {
      this->aSigned = propValue;
    }
    using bSignedTy = ::mlir::BoolAttr;
    bSignedTy bSigned;

    auto getBSigned() const {
      auto &propStorage = this->bSigned;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setBSigned(const ::mlir::BoolAttr &propValue) {
      this->bSigned = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.aSigned == this->aSigned &&
        rhs.bSigned == this->bSigned &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  DotI8OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.dot_i8", odsAttrs.getContext());
  }

  DotI8OpGenericAdaptorBase(DotI8Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::BoolAttr getASignedAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().aSigned);
    return attr;
  }

  bool getASigned();
  ::mlir::BoolAttr getBSignedAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().bSigned);
    return attr;
  }

  bool getBSigned();
};

} // namespace detail
template <typename RangeT>
class DotI8OpGenericAdaptor : public detail::DotI8OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DotI8OpGenericAdaptorBase;
  static constexpr int odsIndex_a = 0;
  static constexpr int odsIndex_b = 1;
  static constexpr int odsIndex_c = 2;
public:
  DotI8OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DotI8OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : DotI8OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DotI8OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DotI8OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  DotI8OpGenericAdaptor(RangeT values, const DotI8OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DotI8Op, typename = std::enable_if_t<std::is_same_v<LateInst, DotI8Op>>>
  DotI8OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getC() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DotI8OpAdaptor : public DotI8OpGenericAdaptor<::mlir::ValueRange> {
public:
  using DotI8OpGenericAdaptor::DotI8OpGenericAdaptor;
  DotI8OpAdaptor(DotI8Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DotI8Op : public ::mlir::Op<DotI8Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::triton::DotOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DotI8OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DotI8OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_a = 0;
  static constexpr int odsIndex_b = 1;
  static constexpr int odsIndex_c = 2;
  static constexpr int odsIndex_d = 0;
  using Properties = FoldAdaptor::Properties;
  using ::mlir::triton::DotOpInterface::Trait<DotI8Op>::verifyDims;
  using ::mlir::triton::DotOpInterface::Trait<DotI8Op>::verifyOutputDims;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("aSigned"), ::llvm::StringRef("bSigned")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getASignedAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getASignedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBSignedAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBSignedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.dot_i8");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getA() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getB() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getC() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getAMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getD() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::BoolAttr getASignedAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().aSigned);
  }

  bool getASigned();
  ::mlir::BoolAttr getBSignedAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().bSigned);
  }

  bool getBSigned();
  void setASignedAttr(::mlir::BoolAttr attr) {
    getProperties().aSigned = attr;
  }

  void setASigned(bool attrValue);
  void setBSignedAttr(::mlir::BoolAttr attr) {
    getProperties().bSigned = attr;
  }

  void setBSigned(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::BoolAttr aSigned, ::mlir::BoolAttr bSigned);
  static DotI8Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::BoolAttr aSigned, ::mlir::BoolAttr bSigned);
  static DotI8Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::BoolAttr aSigned, ::mlir::BoolAttr bSigned);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::BoolAttr aSigned, ::mlir::BoolAttr bSigned);
  static DotI8Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::BoolAttr aSigned, ::mlir::BoolAttr bSigned);
  static DotI8Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::BoolAttr aSigned, ::mlir::BoolAttr bSigned);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, bool aSigned, bool bSigned);
  static DotI8Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, bool aSigned, bool bSigned);
  static DotI8Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, bool aSigned, bool bSigned);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, bool aSigned, bool bSigned);
  static DotI8Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, bool aSigned, bool bSigned);
  static DotI8Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, bool aSigned, bool bSigned);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DotI8Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DotI8Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DotI8Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DotI8Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::DotI8Op)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::DotI8OpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalAssertUniformOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalAssertUniformOpGenericAdaptorBase {
public:
  struct Properties {
    using messageTy = ::mlir::StringAttr;
    messageTy message;

    auto getMessage() const {
      auto &propStorage = this->message;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setMessage(const ::mlir::StringAttr &propValue) {
      this->message = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.message == this->message &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalAssertUniformOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_assert_uniform", odsAttrs.getContext());
  }

  ExperimentalAssertUniformOpGenericAdaptorBase(ExperimentalAssertUniformOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getMessageAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().message);
    return attr;
  }

  ::llvm::StringRef getMessage();
};

} // namespace detail
template <typename RangeT>
class ExperimentalAssertUniformOpGenericAdaptor : public detail::ExperimentalAssertUniformOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalAssertUniformOpGenericAdaptorBase;
  static constexpr int odsIndex_condition = 0;
public:
  ExperimentalAssertUniformOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalAssertUniformOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalAssertUniformOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalAssertUniformOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalAssertUniformOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalAssertUniformOpGenericAdaptor(RangeT values, const ExperimentalAssertUniformOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalAssertUniformOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalAssertUniformOp>>>
  ExperimentalAssertUniformOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCondition() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalAssertUniformOpAdaptor : public ExperimentalAssertUniformOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalAssertUniformOpGenericAdaptor::ExperimentalAssertUniformOpGenericAdaptor;
  ExperimentalAssertUniformOpAdaptor(ExperimentalAssertUniformOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalAssertUniformOp : public ::mlir::Op<ExperimentalAssertUniformOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalAssertUniformOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalAssertUniformOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_condition = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("message")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMessageAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMessageAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_assert_uniform");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCondition() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getConditionMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getMessageAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().message);
  }

  ::llvm::StringRef getMessage();
  void setMessageAttr(::mlir::StringAttr attr) {
    getProperties().message = attr;
  }

  void setMessage(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::StringAttr message);
  static ExperimentalAssertUniformOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value condition, ::mlir::StringAttr message);
  static ExperimentalAssertUniformOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value condition, ::mlir::StringAttr message);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message);
  static ExperimentalAssertUniformOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message);
  static ExperimentalAssertUniformOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::llvm::StringRef message);
  static ExperimentalAssertUniformOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value condition, ::llvm::StringRef message);
  static ExperimentalAssertUniformOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value condition, ::llvm::StringRef message);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message);
  static ExperimentalAssertUniformOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message);
  static ExperimentalAssertUniformOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalAssertUniformOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalAssertUniformOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalAssertUniformOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalAssertUniformOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalAssertUniformOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalAssertUniformOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalBufferDescriptorsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalBufferDescriptorsOpGenericAdaptorBase {
public:
  struct Properties {
    using lengthsTy = ::mlir::DenseI32ArrayAttr;
    lengthsTy lengths;

    auto getLengths() const {
      auto &propStorage = this->lengths;
      return ::llvm::cast<::mlir::DenseI32ArrayAttr>(propStorage);
    }
    void setLengths(const ::mlir::DenseI32ArrayAttr &propValue) {
      this->lengths = propValue;
    }
    using memTypeTy = ::mlir::triton::instrument::MemTypeAttr;
    memTypeTy memType;

    auto getMemType() const {
      auto &propStorage = this->memType;
      return ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(propStorage);
    }
    void setMemType(const ::mlir::triton::instrument::MemTypeAttr &propValue) {
      this->memType = propValue;
    }
    using offsetsTy = ::mlir::DenseI32ArrayAttr;
    offsetsTy offsets;

    auto getOffsets() const {
      auto &propStorage = this->offsets;
      return ::llvm::cast<::mlir::DenseI32ArrayAttr>(propStorage);
    }
    void setOffsets(const ::mlir::DenseI32ArrayAttr &propValue) {
      this->offsets = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.lengths == this->lengths &&
        rhs.memType == this->memType &&
        rhs.offsets == this->offsets &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalBufferDescriptorsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_buffer_descriptors", odsAttrs.getContext());
  }

  ExperimentalBufferDescriptorsOpGenericAdaptorBase(ExperimentalBufferDescriptorsOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DenseI32ArrayAttr getOffsetsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().offsets);
    return attr;
  }

  ::llvm::ArrayRef<int32_t> getOffsets();
  ::mlir::DenseI32ArrayAttr getLengthsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().lengths);
    return attr;
  }

  ::llvm::ArrayRef<int32_t> getLengths();
  ::mlir::triton::instrument::MemTypeAttr getMemTypeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(getProperties().memType);
    return attr;
  }

  ::mlir::triton::instrument::MemType getMemType();
};

} // namespace detail
template <typename RangeT>
class ExperimentalBufferDescriptorsOpGenericAdaptor : public detail::ExperimentalBufferDescriptorsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalBufferDescriptorsOpGenericAdaptorBase;
public:
  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalBufferDescriptorsOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalBufferDescriptorsOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, const ExperimentalBufferDescriptorsOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalBufferDescriptorsOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalBufferDescriptorsOp>>>
  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalBufferDescriptorsOpAdaptor : public ExperimentalBufferDescriptorsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalBufferDescriptorsOpGenericAdaptor::ExperimentalBufferDescriptorsOpGenericAdaptor;
  ExperimentalBufferDescriptorsOpAdaptor(ExperimentalBufferDescriptorsOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalBufferDescriptorsOp : public ::mlir::Op<ExperimentalBufferDescriptorsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalBufferDescriptorsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalBufferDescriptorsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lengths"), ::llvm::StringRef("memType"), ::llvm::StringRef("offsets")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLengthsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLengthsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOffsetsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_buffer_descriptors");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseI32ArrayAttr getOffsetsAttr() {
    return ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().offsets);
  }

  ::llvm::ArrayRef<int32_t> getOffsets();
  ::mlir::DenseI32ArrayAttr getLengthsAttr() {
    return ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().lengths);
  }

  ::llvm::ArrayRef<int32_t> getLengths();
  ::mlir::triton::instrument::MemTypeAttr getMemTypeAttr() {
    return ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(getProperties().memType);
  }

  ::mlir::triton::instrument::MemType getMemType();
  void setOffsetsAttr(::mlir::DenseI32ArrayAttr attr) {
    getProperties().offsets = attr;
  }

  void setOffsets(::llvm::ArrayRef<int32_t> attrValue);
  void setLengthsAttr(::mlir::DenseI32ArrayAttr attr) {
    getProperties().lengths = attr;
  }

  void setLengths(::llvm::ArrayRef<int32_t> attrValue);
  void setMemTypeAttr(::mlir::triton::instrument::MemTypeAttr attr) {
    getProperties().memType = attr;
  }

  void setMemType(::mlir::triton::instrument::MemType attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalBufferDescriptorsOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalBufferDescriptorsOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalClusterCTAIdOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalClusterCTAIdOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalClusterCTAIdOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_cluster_cta_id", odsAttrs.getContext());
  }

  ExperimentalClusterCTAIdOpGenericAdaptorBase(ExperimentalClusterCTAIdOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalClusterCTAIdOpGenericAdaptor : public detail::ExperimentalClusterCTAIdOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalClusterCTAIdOpGenericAdaptorBase;
public:
  ExperimentalClusterCTAIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalClusterCTAIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalClusterCTAIdOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalClusterCTAIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalClusterCTAIdOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalClusterCTAIdOpGenericAdaptor(RangeT values, const ExperimentalClusterCTAIdOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalClusterCTAIdOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalClusterCTAIdOp>>>
  ExperimentalClusterCTAIdOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalClusterCTAIdOpAdaptor : public ExperimentalClusterCTAIdOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalClusterCTAIdOpGenericAdaptor::ExperimentalClusterCTAIdOpGenericAdaptor;
  ExperimentalClusterCTAIdOpAdaptor(ExperimentalClusterCTAIdOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalClusterCTAIdOp : public ::mlir::Op<ExperimentalClusterCTAIdOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalClusterCTAIdOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalClusterCTAIdOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_cluster_cta_id");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result);
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result);
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::PropertyRef properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalClusterCTAIdOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalFPSanEmbedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalFPSanEmbedOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalFPSanEmbedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_fpsan_embed", odsAttrs.getContext());
  }

  ExperimentalFPSanEmbedOpGenericAdaptorBase(ExperimentalFPSanEmbedOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalFPSanEmbedOpGenericAdaptor : public detail::ExperimentalFPSanEmbedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalFPSanEmbedOpGenericAdaptorBase;
  static constexpr int odsIndex_val = 0;
public:
  ExperimentalFPSanEmbedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalFPSanEmbedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalFPSanEmbedOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalFPSanEmbedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalFPSanEmbedOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalFPSanEmbedOpGenericAdaptor(RangeT values, const ExperimentalFPSanEmbedOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalFPSanEmbedOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalFPSanEmbedOp>>>
  ExperimentalFPSanEmbedOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVal() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalFPSanEmbedOpAdaptor : public ExperimentalFPSanEmbedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalFPSanEmbedOpGenericAdaptor::ExperimentalFPSanEmbedOpGenericAdaptor;
  ExperimentalFPSanEmbedOpAdaptor(ExperimentalFPSanEmbedOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalFPSanEmbedOp : public ::mlir::Op<ExperimentalFPSanEmbedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::Elementwise, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultEncoding> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalFPSanEmbedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalFPSanEmbedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_val = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_fpsan_embed");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getValMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value val);
  static ExperimentalFPSanEmbedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value val);
  static ExperimentalFPSanEmbedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static ExperimentalFPSanEmbedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static ExperimentalFPSanEmbedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalFPSanEmbedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalFPSanEmbedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalFPSanEmbedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalFPSanEmbedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalFPSanEmbedOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalFPSanUnembedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalFPSanUnembedOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalFPSanUnembedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_fpsan_unembed", odsAttrs.getContext());
  }

  ExperimentalFPSanUnembedOpGenericAdaptorBase(ExperimentalFPSanUnembedOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalFPSanUnembedOpGenericAdaptor : public detail::ExperimentalFPSanUnembedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalFPSanUnembedOpGenericAdaptorBase;
  static constexpr int odsIndex_val = 0;
public:
  ExperimentalFPSanUnembedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalFPSanUnembedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalFPSanUnembedOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalFPSanUnembedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalFPSanUnembedOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalFPSanUnembedOpGenericAdaptor(RangeT values, const ExperimentalFPSanUnembedOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalFPSanUnembedOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalFPSanUnembedOp>>>
  ExperimentalFPSanUnembedOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVal() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalFPSanUnembedOpAdaptor : public ExperimentalFPSanUnembedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalFPSanUnembedOpGenericAdaptor::ExperimentalFPSanUnembedOpGenericAdaptor;
  ExperimentalFPSanUnembedOpAdaptor(ExperimentalFPSanUnembedOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalFPSanUnembedOp : public ::mlir::Op<ExperimentalFPSanUnembedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::Elementwise, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultEncoding> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalFPSanUnembedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalFPSanUnembedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_val = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_fpsan_unembed");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getValMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value val);
  static ExperimentalFPSanUnembedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value val);
  static ExperimentalFPSanUnembedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static ExperimentalFPSanUnembedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static ExperimentalFPSanUnembedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalFPSanUnembedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalFPSanUnembedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalFPSanUnembedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalFPSanUnembedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalFPSanUnembedOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanAtomicCASOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanAtomicCASOpGenericAdaptorBase {
public:
  struct Properties {
    using scopeTy = ::mlir::triton::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanAtomicCASOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_atomic_cas", odsAttrs.getContext());
  }

  ExperimentalGSanAtomicCASOpGenericAdaptorBase(ExperimentalGSanAtomicCASOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanAtomicCASOpGenericAdaptor : public detail::ExperimentalGSanAtomicCASOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanAtomicCASOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_cmp = 1;
  static constexpr int odsIndex_val = 2;
public:
  ExperimentalGSanAtomicCASOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanAtomicCASOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanAtomicCASOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanAtomicCASOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanAtomicCASOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanAtomicCASOpGenericAdaptor(RangeT values, const ExperimentalGSanAtomicCASOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanAtomicCASOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanAtomicCASOp>>>
  ExperimentalGSanAtomicCASOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getCmp() {
    return (*getODSOperands(1).begin());
  }

  ValueT getVal() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanAtomicCASOpAdaptor : public ExperimentalGSanAtomicCASOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanAtomicCASOpGenericAdaptor::ExperimentalGSanAtomicCASOpGenericAdaptor;
  ExperimentalGSanAtomicCASOpAdaptor(ExperimentalGSanAtomicCASOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanAtomicCASOp : public ::mlir::Op<ExperimentalGSanAtomicCASOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::AtomicOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultEncoding, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanAtomicCASOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanAtomicCASOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_cmp = 1;
  static constexpr int odsIndex_val = 2;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_atomic_cas");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getCmp() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCmpMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getValMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::MemSyncScope getScope();
  void setSemAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicCASOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicCASOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicCASOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicCASOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicCASOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanAtomicCASOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanAtomicCASOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanAtomicLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanAtomicLoadOpGenericAdaptorBase {
public:
  struct Properties {
    using scopeTy = ::mlir::triton::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanAtomicLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_atomic_load", odsAttrs.getContext());
  }

  ExperimentalGSanAtomicLoadOpGenericAdaptorBase(ExperimentalGSanAtomicLoadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanAtomicLoadOpGenericAdaptor : public detail::ExperimentalGSanAtomicLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanAtomicLoadOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_mask = 1;
public:
  ExperimentalGSanAtomicLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanAtomicLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanAtomicLoadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanAtomicLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanAtomicLoadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanAtomicLoadOpGenericAdaptor(RangeT values, const ExperimentalGSanAtomicLoadOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanAtomicLoadOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanAtomicLoadOp>>>
  ExperimentalGSanAtomicLoadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanAtomicLoadOpAdaptor : public ExperimentalGSanAtomicLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanAtomicLoadOpGenericAdaptor::ExperimentalGSanAtomicLoadOpGenericAdaptor;
  ExperimentalGSanAtomicLoadOpAdaptor(ExperimentalGSanAtomicLoadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanAtomicLoadOp : public ::mlir::Op<ExperimentalGSanAtomicLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::AtomicOpInterface::Trait, ::mlir::OpTrait::SameLoadStoreOperandsAndResultShape, ::mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding, ::mlir::InferTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanAtomicLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanAtomicLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_mask = 1;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_atomic_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMask() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::MemSyncScope getScope();
  void setSemAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::PropertyRef properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanAtomicLoadOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanAtomicLoadOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanAtomicPollOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanAtomicPollOpGenericAdaptorBase {
public:
  struct Properties {
    using scopeTy = ::mlir::triton::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanAtomicPollOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_atomic_poll", odsAttrs.getContext());
  }

  ExperimentalGSanAtomicPollOpGenericAdaptorBase(ExperimentalGSanAtomicPollOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanAtomicPollOpGenericAdaptor : public detail::ExperimentalGSanAtomicPollOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanAtomicPollOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_matched = 1;
public:
  ExperimentalGSanAtomicPollOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanAtomicPollOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanAtomicPollOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanAtomicPollOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanAtomicPollOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanAtomicPollOpGenericAdaptor(RangeT values, const ExperimentalGSanAtomicPollOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanAtomicPollOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanAtomicPollOp>>>
  ExperimentalGSanAtomicPollOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMatched() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanAtomicPollOpAdaptor : public ExperimentalGSanAtomicPollOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanAtomicPollOpGenericAdaptor::ExperimentalGSanAtomicPollOpGenericAdaptor;
  ExperimentalGSanAtomicPollOpAdaptor(ExperimentalGSanAtomicPollOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanAtomicPollOp : public ::mlir::Op<ExperimentalGSanAtomicPollOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::SameOperandsEncoding, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanAtomicPollOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanAtomicPollOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_matched = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_atomic_poll");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMatched() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getMatchedMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::MemSyncScope getScope();
  void setSemAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicPollOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value matched, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicPollOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicPollOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicPollOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicPollOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanAtomicPollOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanAtomicPollOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanAtomicRMWOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanAtomicRMWOpGenericAdaptorBase {
public:
  struct Properties {
    using atomic_rmw_opTy = ::mlir::triton::RMWOpAttr;
    atomic_rmw_opTy atomic_rmw_op;

    auto getAtomicRmwOp() const {
      auto &propStorage = this->atomic_rmw_op;
      return ::llvm::cast<::mlir::triton::RMWOpAttr>(propStorage);
    }
    void setAtomicRmwOp(const ::mlir::triton::RMWOpAttr &propValue) {
      this->atomic_rmw_op = propValue;
    }
    using scopeTy = ::mlir::triton::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.atomic_rmw_op == this->atomic_rmw_op &&
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanAtomicRMWOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_atomic_rmw", odsAttrs.getContext());
  }

  ExperimentalGSanAtomicRMWOpGenericAdaptorBase(ExperimentalGSanAtomicRMWOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::RMWOpAttr getAtomicRmwOpAttr() {
    auto attr = ::llvm::cast<::mlir::triton::RMWOpAttr>(getProperties().atomic_rmw_op);
    return attr;
  }

  ::mlir::triton::RMWOp getAtomicRmwOp();
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanAtomicRMWOpGenericAdaptor : public detail::ExperimentalGSanAtomicRMWOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanAtomicRMWOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_val = 1;
  static constexpr int odsIndex_mask = 2;
public:
  ExperimentalGSanAtomicRMWOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanAtomicRMWOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanAtomicRMWOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanAtomicRMWOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanAtomicRMWOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanAtomicRMWOpGenericAdaptor(RangeT values, const ExperimentalGSanAtomicRMWOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanAtomicRMWOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanAtomicRMWOp>>>
  ExperimentalGSanAtomicRMWOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getVal() {
    return (*getODSOperands(1).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanAtomicRMWOpAdaptor : public ExperimentalGSanAtomicRMWOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanAtomicRMWOpGenericAdaptor::ExperimentalGSanAtomicRMWOpGenericAdaptor;
  ExperimentalGSanAtomicRMWOpAdaptor(ExperimentalGSanAtomicRMWOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanAtomicRMWOp : public ::mlir::Op<ExperimentalGSanAtomicRMWOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::AtomicOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultEncoding, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanAtomicRMWOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanAtomicRMWOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_val = 1;
  static constexpr int odsIndex_mask = 2;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("atomic_rmw_op"), ::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAtomicRmwOpAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAtomicRmwOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_atomic_rmw");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getValMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::RMWOpAttr getAtomicRmwOpAttr() {
    return ::llvm::cast<::mlir::triton::RMWOpAttr>(getProperties().atomic_rmw_op);
  }

  ::mlir::triton::RMWOp getAtomicRmwOp();
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::MemSyncScope getScope();
  void setAtomicRmwOpAttr(::mlir::triton::RMWOpAttr attr) {
    getProperties().atomic_rmw_op = attr;
  }

  void setAtomicRmwOp(::mlir::triton::RMWOp attrValue);
  void setSemAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::triton::RMWOpAttr atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::triton::RMWOpAttr atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::triton::RMWOpAttr atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::RMWOpAttr atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::triton::RMWOpAttr atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::triton::RMWOpAttr atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::triton::RMWOp atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::triton::RMWOp atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::triton::RMWOp atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::RMWOp atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::triton::RMWOp atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicRMWOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::triton::RMWOp atomic_rmw_op, ::mlir::Value ptr, ::mlir::Value val, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicRMWOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicRMWOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicRMWOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicRMWOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanAtomicRMWOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanAtomicRMWOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanAtomicStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanAtomicStoreOpGenericAdaptorBase {
public:
  struct Properties {
    using scopeTy = ::mlir::triton::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanAtomicStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_atomic_store", odsAttrs.getContext());
  }

  ExperimentalGSanAtomicStoreOpGenericAdaptorBase(ExperimentalGSanAtomicStoreOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanAtomicStoreOpGenericAdaptor : public detail::ExperimentalGSanAtomicStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanAtomicStoreOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_value = 1;
  static constexpr int odsIndex_mask = 2;
public:
  ExperimentalGSanAtomicStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanAtomicStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanAtomicStoreOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanAtomicStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanAtomicStoreOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanAtomicStoreOpGenericAdaptor(RangeT values, const ExperimentalGSanAtomicStoreOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanAtomicStoreOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanAtomicStoreOp>>>
  ExperimentalGSanAtomicStoreOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getValue() {
    return (*getODSOperands(1).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanAtomicStoreOpAdaptor : public ExperimentalGSanAtomicStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanAtomicStoreOpGenericAdaptor::ExperimentalGSanAtomicStoreOpGenericAdaptor;
  ExperimentalGSanAtomicStoreOpAdaptor(ExperimentalGSanAtomicStoreOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanAtomicStoreOp : public ::mlir::Op<ExperimentalGSanAtomicStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::AtomicOpInterface::Trait, ::mlir::OpTrait::SameLoadStoreOperandsShape, ::mlir::OpTrait::SameLoadStoreOperandsEncoding, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanAtomicStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanAtomicStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_value = 1;
  static constexpr int odsIndex_mask = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_atomic_store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getValueMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::MemSyncScope getScope();
  void setSemAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicStoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicStoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicStoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicStoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicStoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanAtomicStoreOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanAtomicStoreOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanAtomicTensorDescAccessOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase {
public:
  struct Properties {
    using scopeTy = ::mlir::triton::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_atomic_tensordesc_access", odsAttrs.getContext());
  }

  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase(ExperimentalGSanAtomicTensorDescAccessOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor : public detail::ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_coords = 1;
public:
  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(RangeT values, const ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanAtomicTensorDescAccessOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanAtomicTensorDescAccessOp>>>
  ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDesc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getCoords() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanAtomicTensorDescAccessOpAdaptor : public ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor::ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor;
  ExperimentalGSanAtomicTensorDescAccessOpAdaptor(ExperimentalGSanAtomicTensorDescAccessOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanAtomicTensorDescAccessOp : public ::mlir::Op<ExperimentalGSanAtomicTensorDescAccessOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanAtomicTensorDescAccessOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_coords = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_atomic_tensordesc_access");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::TensorDescInterface> getDesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::TensorDescInterface>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getCoords() {
    return getODSOperands(1);
  }

  ::mlir::OpOperand &getDescMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getCoordsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::MemSemantic getSem();
  ::mlir::triton::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::MemSyncScope getScope();
  void setSemAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemanticAttr sem, ::mlir::triton::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, ::mlir::triton::MemSemantic sem, ::mlir::triton::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanAtomicTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanAtomicTensorDescAccessOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanAtomicTensorDescAccessOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanClusterBarrierInitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanClusterBarrierInitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanClusterBarrierInitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_cluster_barrier_init", odsAttrs.getContext());
  }

  ExperimentalGSanClusterBarrierInitOpGenericAdaptorBase(ExperimentalGSanClusterBarrierInitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanClusterBarrierInitOpGenericAdaptor : public detail::ExperimentalGSanClusterBarrierInitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanClusterBarrierInitOpGenericAdaptorBase;
  static constexpr int odsIndex_scratch = 0;
public:
  ExperimentalGSanClusterBarrierInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanClusterBarrierInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanClusterBarrierInitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanClusterBarrierInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanClusterBarrierInitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanClusterBarrierInitOpGenericAdaptor(RangeT values, const ExperimentalGSanClusterBarrierInitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanClusterBarrierInitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanClusterBarrierInitOp>>>
  ExperimentalGSanClusterBarrierInitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScratch() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanClusterBarrierInitOpAdaptor : public ExperimentalGSanClusterBarrierInitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanClusterBarrierInitOpGenericAdaptor::ExperimentalGSanClusterBarrierInitOpGenericAdaptor;
  ExperimentalGSanClusterBarrierInitOpAdaptor(ExperimentalGSanClusterBarrierInitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanClusterBarrierInitOp : public ::mlir::Op<ExperimentalGSanClusterBarrierInitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanClusterBarrierInitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanClusterBarrierInitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_scratch = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_cluster_barrier_init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getScratch() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getScratchMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanClusterBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanClusterBarrierInitOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanClusterBarrierSyncOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanClusterBarrierSyncOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanClusterBarrierSyncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_cluster_barrier_sync", odsAttrs.getContext());
  }

  ExperimentalGSanClusterBarrierSyncOpGenericAdaptorBase(ExperimentalGSanClusterBarrierSyncOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanClusterBarrierSyncOpGenericAdaptor : public detail::ExperimentalGSanClusterBarrierSyncOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanClusterBarrierSyncOpGenericAdaptorBase;
  static constexpr int odsIndex_scratch = 0;
public:
  ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(RangeT values, const ExperimentalGSanClusterBarrierSyncOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanClusterBarrierSyncOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanClusterBarrierSyncOp>>>
  ExperimentalGSanClusterBarrierSyncOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScratch() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanClusterBarrierSyncOpAdaptor : public ExperimentalGSanClusterBarrierSyncOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanClusterBarrierSyncOpGenericAdaptor::ExperimentalGSanClusterBarrierSyncOpGenericAdaptor;
  ExperimentalGSanClusterBarrierSyncOpAdaptor(ExperimentalGSanClusterBarrierSyncOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanClusterBarrierSyncOp : public ::mlir::Op<ExperimentalGSanClusterBarrierSyncOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanClusterBarrierSyncOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanClusterBarrierSyncOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_scratch = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_cluster_barrier_sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getScratch() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getScratchMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch);
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanClusterBarrierSyncOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanClusterBarrierSyncOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanGridDependencyWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanGridDependencyWaitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanGridDependencyWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_grid_dependency_wait", odsAttrs.getContext());
  }

  ExperimentalGSanGridDependencyWaitOpGenericAdaptorBase(ExperimentalGSanGridDependencyWaitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanGridDependencyWaitOpGenericAdaptor : public detail::ExperimentalGSanGridDependencyWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanGridDependencyWaitOpGenericAdaptorBase;
public:
  ExperimentalGSanGridDependencyWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanGridDependencyWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanGridDependencyWaitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanGridDependencyWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanGridDependencyWaitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanGridDependencyWaitOpGenericAdaptor(RangeT values, const ExperimentalGSanGridDependencyWaitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanGridDependencyWaitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanGridDependencyWaitOp>>>
  ExperimentalGSanGridDependencyWaitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanGridDependencyWaitOpAdaptor : public ExperimentalGSanGridDependencyWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanGridDependencyWaitOpGenericAdaptor::ExperimentalGSanGridDependencyWaitOpGenericAdaptor;
  ExperimentalGSanGridDependencyWaitOpAdaptor(ExperimentalGSanGridDependencyWaitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanGridDependencyWaitOp : public ::mlir::Op<ExperimentalGSanGridDependencyWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanGridDependencyWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanGridDependencyWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_grid_dependency_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanGridDependencyWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanGridDependencyWaitOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanIndexedTensorDescAccessOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase {
public:
  struct Properties {
    using isStoreTy = ::mlir::BoolAttr;
    isStoreTy isStore;

    auto getIsStore() const {
      auto &propStorage = this->isStore;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setIsStore(const ::mlir::BoolAttr &propValue) {
      this->isStore = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.isStore == this->isStore &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_indexed_tensordesc_access", odsAttrs.getContext());
  }

  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase(ExperimentalGSanIndexedTensorDescAccessOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::BoolAttr getIsStoreAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().isStore);
    return attr;
  }

  bool getIsStore();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor : public detail::ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_indices = 1;
  static constexpr int odsIndex_offset = 2;
  static constexpr int odsIndex_pred = 3;
public:
  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(RangeT values, const ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanIndexedTensorDescAccessOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanIndexedTensorDescAccessOp>>>
  ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDesc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIndices() {
    return (*getODSOperands(1).begin());
  }

  ValueT getOffset() {
    return (*getODSOperands(2).begin());
  }

  ValueT getPred() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanIndexedTensorDescAccessOpAdaptor : public ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor::ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor;
  ExperimentalGSanIndexedTensorDescAccessOpAdaptor(ExperimentalGSanIndexedTensorDescAccessOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanIndexedTensorDescAccessOp : public ::mlir::Op<ExperimentalGSanIndexedTensorDescAccessOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanIndexedTensorDescAccessOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_indices = 1;
  static constexpr int odsIndex_offset = 2;
  static constexpr int odsIndex_pred = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isStore")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsStoreAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsStoreAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_indexed_tensordesc_access");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::TensorDescInterface> getDesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::TensorDescInterface>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getIndices() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getOffset() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getDescMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIndicesMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOffsetMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getPredMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::BoolAttr getIsStoreAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().isStore);
  }

  bool getIsStore();
  void setIsStoreAttr(::mlir::BoolAttr attr) {
    getProperties().isStore = attr;
  }

  void setIsStore(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, bool isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value indices, ::mlir::Value offset, /*optional*/::mlir::Value pred, bool isStore);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanIndexedTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanIndexedTensorDescAccessOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanIndexedTensorDescAccessOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanInitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanInitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanInitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_init", odsAttrs.getContext());
  }

  ExperimentalGSanInitOpGenericAdaptorBase(ExperimentalGSanInitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanInitOpGenericAdaptor : public detail::ExperimentalGSanInitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanInitOpGenericAdaptorBase;
public:
  ExperimentalGSanInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanInitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanInitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanInitOpGenericAdaptor(RangeT values, const ExperimentalGSanInitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanInitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanInitOp>>>
  ExperimentalGSanInitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanInitOpAdaptor : public ExperimentalGSanInitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanInitOpGenericAdaptor::ExperimentalGSanInitOpGenericAdaptor;
  ExperimentalGSanInitOpAdaptor(ExperimentalGSanInitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanInitOp : public ::mlir::Op<ExperimentalGSanInitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanInitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanInitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ExperimentalGSanInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static ExperimentalGSanInitOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static ExperimentalGSanInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static ExperimentalGSanInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanInitOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanKernelExitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanKernelExitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanKernelExitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_kernel_exit", odsAttrs.getContext());
  }

  ExperimentalGSanKernelExitOpGenericAdaptorBase(ExperimentalGSanKernelExitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanKernelExitOpGenericAdaptor : public detail::ExperimentalGSanKernelExitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanKernelExitOpGenericAdaptorBase;
public:
  ExperimentalGSanKernelExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanKernelExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanKernelExitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanKernelExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanKernelExitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanKernelExitOpGenericAdaptor(RangeT values, const ExperimentalGSanKernelExitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanKernelExitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanKernelExitOp>>>
  ExperimentalGSanKernelExitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanKernelExitOpAdaptor : public ExperimentalGSanKernelExitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanKernelExitOpGenericAdaptor::ExperimentalGSanKernelExitOpGenericAdaptor;
  ExperimentalGSanKernelExitOpAdaptor(ExperimentalGSanKernelExitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanKernelExitOp : public ::mlir::Op<ExperimentalGSanKernelExitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanKernelExitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanKernelExitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_kernel_exit");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ExperimentalGSanKernelExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static ExperimentalGSanKernelExitOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static ExperimentalGSanKernelExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static ExperimentalGSanKernelExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanKernelExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanKernelExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanKernelExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanKernelExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanKernelExitOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanMBarrierArriveOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanMBarrierArriveOpGenericAdaptorBase {
public:
  struct Properties {
    using countTy = ::mlir::IntegerAttr;
    countTy count;

    auto getCount() const {
      auto &propStorage = this->count;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setCount(const ::mlir::IntegerAttr &propValue) {
      this->count = propValue;
    }
    using multicastTy = ::mlir::BoolAttr;
    multicastTy multicast;

    auto getMulticast() const {
      auto &propStorage = this->multicast;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setMulticast(const ::mlir::BoolAttr &propValue) {
      this->multicast = propValue;
    }
    using multicastMasksTy = ::mlir::DenseI32ArrayAttr;
    multicastMasksTy multicastMasks;

    auto getMulticastMasks() const {
      auto &propStorage = this->multicastMasks;
      return ::llvm::cast<::mlir::DenseI32ArrayAttr>(propStorage);
    }
    void setMulticastMasks(const ::mlir::DenseI32ArrayAttr &propValue) {
      this->multicastMasks = propValue;
    }
    using publishClockTy = ::mlir::BoolAttr;
    publishClockTy publishClock;

    auto getPublishClock() const {
      auto &propStorage = this->publishClock;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setPublishClock(const ::mlir::BoolAttr &propValue) {
      this->publishClock = propValue;
    }
    using sourceBroadcastMaskTy = ::mlir::IntegerAttr;
    sourceBroadcastMaskTy sourceBroadcastMask;

    auto getSourceBroadcastMask() const {
      auto &propStorage = this->sourceBroadcastMask;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setSourceBroadcastMask(const ::mlir::IntegerAttr &propValue) {
      this->sourceBroadcastMask = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.count == this->count &&
        rhs.multicast == this->multicast &&
        rhs.multicastMasks == this->multicastMasks &&
        rhs.publishClock == this->publishClock &&
        rhs.sourceBroadcastMask == this->sourceBroadcastMask &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanMBarrierArriveOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_mbarrier_arrive", odsAttrs.getContext());
  }

  ExperimentalGSanMBarrierArriveOpGenericAdaptorBase(ExperimentalGSanMBarrierArriveOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getCountAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().count);
    return attr;
  }

  uint32_t getCount();
  ::mlir::DenseI32ArrayAttr getMulticastMasksAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().multicastMasks);
    return attr;
  }

  ::llvm::ArrayRef<int32_t> getMulticastMasks();
  ::mlir::BoolAttr getMulticastAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().multicast);
    return attr;
  }

  bool getMulticast();
  ::mlir::IntegerAttr getSourceBroadcastMaskAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().sourceBroadcastMask);
    return attr;
  }

  uint32_t getSourceBroadcastMask();
  ::mlir::BoolAttr getPublishClockAttr();
  bool getPublishClock();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanMBarrierArriveOpGenericAdaptor : public detail::ExperimentalGSanMBarrierArriveOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanMBarrierArriveOpGenericAdaptorBase;
  static constexpr int odsIndex_scratch = 0;
  static constexpr int odsIndex_barrier = 1;
  static constexpr int odsIndex_pred = 2;
public:
  ExperimentalGSanMBarrierArriveOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanMBarrierArriveOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanMBarrierArriveOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanMBarrierArriveOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanMBarrierArriveOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanMBarrierArriveOpGenericAdaptor(RangeT values, const ExperimentalGSanMBarrierArriveOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanMBarrierArriveOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanMBarrierArriveOp>>>
  ExperimentalGSanMBarrierArriveOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScratch() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBarrier() {
    return (*getODSOperands(1).begin());
  }

  ValueT getPred() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanMBarrierArriveOpAdaptor : public ExperimentalGSanMBarrierArriveOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanMBarrierArriveOpGenericAdaptor::ExperimentalGSanMBarrierArriveOpGenericAdaptor;
  ExperimentalGSanMBarrierArriveOpAdaptor(ExperimentalGSanMBarrierArriveOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanMBarrierArriveOp : public ::mlir::Op<ExperimentalGSanMBarrierArriveOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanMBarrierArriveOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanMBarrierArriveOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_scratch = 0;
  static constexpr int odsIndex_barrier = 1;
  static constexpr int odsIndex_pred = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("count"), ::llvm::StringRef("multicast"), ::llvm::StringRef("multicastMasks"), ::llvm::StringRef("publishClock"), ::llvm::StringRef("sourceBroadcastMask")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCountAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMulticastAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMulticastAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMulticastMasksAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMulticastMasksAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getPublishClockAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getPublishClockAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSourceBroadcastMaskAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSourceBroadcastMaskAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_mbarrier_arrive");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getScratch() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getScratchMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBarrierMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPredMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getCountAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().count);
  }

  uint32_t getCount();
  ::mlir::DenseI32ArrayAttr getMulticastMasksAttr() {
    return ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().multicastMasks);
  }

  ::llvm::ArrayRef<int32_t> getMulticastMasks();
  ::mlir::BoolAttr getMulticastAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().multicast);
  }

  bool getMulticast();
  ::mlir::IntegerAttr getSourceBroadcastMaskAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().sourceBroadcastMask);
  }

  uint32_t getSourceBroadcastMask();
  ::mlir::BoolAttr getPublishClockAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().publishClock);
  }

  bool getPublishClock();
  void setCountAttr(::mlir::IntegerAttr attr) {
    getProperties().count = attr;
  }

  void setCount(uint32_t attrValue);
  void setMulticastMasksAttr(::mlir::DenseI32ArrayAttr attr) {
    getProperties().multicastMasks = attr;
  }

  void setMulticastMasks(::llvm::ArrayRef<int32_t> attrValue);
  void setMulticastAttr(::mlir::BoolAttr attr) {
    getProperties().multicast = attr;
  }

  void setMulticast(bool attrValue);
  void setSourceBroadcastMaskAttr(::mlir::IntegerAttr attr) {
    getProperties().sourceBroadcastMask = attr;
  }

  void setSourceBroadcastMask(uint32_t attrValue);
  void setPublishClockAttr(::mlir::BoolAttr attr) {
    getProperties().publishClock = attr;
  }

  void setPublishClock(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, ::mlir::IntegerAttr count, ::mlir::DenseI32ArrayAttr multicastMasks, ::mlir::BoolAttr multicast, ::mlir::IntegerAttr sourceBroadcastMask, ::mlir::BoolAttr publishClock = nullptr);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, ::mlir::IntegerAttr count, ::mlir::DenseI32ArrayAttr multicastMasks, ::mlir::BoolAttr multicast, ::mlir::IntegerAttr sourceBroadcastMask, ::mlir::BoolAttr publishClock = nullptr);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, ::mlir::IntegerAttr count, ::mlir::DenseI32ArrayAttr multicastMasks, ::mlir::BoolAttr multicast, ::mlir::IntegerAttr sourceBroadcastMask, ::mlir::BoolAttr publishClock = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, ::mlir::IntegerAttr count, ::mlir::DenseI32ArrayAttr multicastMasks, ::mlir::BoolAttr multicast, ::mlir::IntegerAttr sourceBroadcastMask, ::mlir::BoolAttr publishClock = nullptr);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, ::mlir::IntegerAttr count, ::mlir::DenseI32ArrayAttr multicastMasks, ::mlir::BoolAttr multicast, ::mlir::IntegerAttr sourceBroadcastMask, ::mlir::BoolAttr publishClock = nullptr);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, ::mlir::IntegerAttr count, ::mlir::DenseI32ArrayAttr multicastMasks, ::mlir::BoolAttr multicast, ::mlir::IntegerAttr sourceBroadcastMask, ::mlir::BoolAttr publishClock = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, uint32_t count, ::llvm::ArrayRef<int32_t> multicastMasks, bool multicast, uint32_t sourceBroadcastMask, bool publishClock = true);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, uint32_t count, ::llvm::ArrayRef<int32_t> multicastMasks, bool multicast, uint32_t sourceBroadcastMask, bool publishClock = true);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, uint32_t count, ::llvm::ArrayRef<int32_t> multicastMasks, bool multicast, uint32_t sourceBroadcastMask, bool publishClock = true);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, uint32_t count, ::llvm::ArrayRef<int32_t> multicastMasks, bool multicast, uint32_t sourceBroadcastMask, bool publishClock = true);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, uint32_t count, ::llvm::ArrayRef<int32_t> multicastMasks, bool multicast, uint32_t sourceBroadcastMask, bool publishClock = true);
  static ExperimentalGSanMBarrierArriveOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value pred, uint32_t count, ::llvm::ArrayRef<int32_t> multicastMasks, bool multicast, uint32_t sourceBroadcastMask, bool publishClock = true);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierArriveOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierArriveOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierArriveOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierArriveOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanMBarrierArriveOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanMBarrierArriveOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanMBarrierInitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanMBarrierInitOpGenericAdaptorBase {
public:
  struct Properties {
    using expectedCountTy = ::mlir::IntegerAttr;
    expectedCountTy expectedCount;

    auto getExpectedCount() const {
      auto &propStorage = this->expectedCount;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setExpectedCount(const ::mlir::IntegerAttr &propValue) {
      this->expectedCount = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.expectedCount == this->expectedCount &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanMBarrierInitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_mbarrier_init", odsAttrs.getContext());
  }

  ExperimentalGSanMBarrierInitOpGenericAdaptorBase(ExperimentalGSanMBarrierInitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getExpectedCountAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().expectedCount);
    return attr;
  }

  uint32_t getExpectedCount();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanMBarrierInitOpGenericAdaptor : public detail::ExperimentalGSanMBarrierInitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanMBarrierInitOpGenericAdaptorBase;
  static constexpr int odsIndex_scratch = 0;
  static constexpr int odsIndex_barrier = 1;
public:
  ExperimentalGSanMBarrierInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanMBarrierInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanMBarrierInitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanMBarrierInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanMBarrierInitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanMBarrierInitOpGenericAdaptor(RangeT values, const ExperimentalGSanMBarrierInitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanMBarrierInitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanMBarrierInitOp>>>
  ExperimentalGSanMBarrierInitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScratch() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBarrier() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanMBarrierInitOpAdaptor : public ExperimentalGSanMBarrierInitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanMBarrierInitOpGenericAdaptor::ExperimentalGSanMBarrierInitOpGenericAdaptor;
  ExperimentalGSanMBarrierInitOpAdaptor(ExperimentalGSanMBarrierInitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanMBarrierInitOp : public ::mlir::Op<ExperimentalGSanMBarrierInitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanMBarrierInitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanMBarrierInitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_scratch = 0;
  static constexpr int odsIndex_barrier = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("expectedCount")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getExpectedCountAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getExpectedCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_mbarrier_init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getScratch() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getScratchMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBarrierMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getExpectedCountAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().expectedCount);
  }

  uint32_t getExpectedCount();
  void setExpectedCountAttr(::mlir::IntegerAttr attr) {
    getProperties().expectedCount = attr;
  }

  void setExpectedCount(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::IntegerAttr expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::IntegerAttr expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::IntegerAttr expectedCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::IntegerAttr expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::IntegerAttr expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::IntegerAttr expectedCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, ::mlir::Value barrier, uint32_t expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, ::mlir::Value barrier, uint32_t expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, ::mlir::Value barrier, uint32_t expectedCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, uint32_t expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, uint32_t expectedCount);
  static ExperimentalGSanMBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, uint32_t expectedCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanMBarrierInitOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanMBarrierInitOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanMBarrierTableInitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase {
public:
  struct Properties {
    using capacityTy = ::mlir::IntegerAttr;
    capacityTy capacity;

    auto getCapacity() const {
      auto &propStorage = this->capacity;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setCapacity(const ::mlir::IntegerAttr &propValue) {
      this->capacity = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.capacity == this->capacity &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_mbarrier_table_init", odsAttrs.getContext());
  }

  ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase(ExperimentalGSanMBarrierTableInitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getCapacityAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().capacity);
    return attr;
  }

  uint32_t getCapacity();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanMBarrierTableInitOpGenericAdaptor : public detail::ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase;
  static constexpr int odsIndex_scratch = 0;
public:
  ExperimentalGSanMBarrierTableInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanMBarrierTableInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanMBarrierTableInitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanMBarrierTableInitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanMBarrierTableInitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanMBarrierTableInitOpGenericAdaptor(RangeT values, const ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanMBarrierTableInitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanMBarrierTableInitOp>>>
  ExperimentalGSanMBarrierTableInitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScratch() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanMBarrierTableInitOpAdaptor : public ExperimentalGSanMBarrierTableInitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanMBarrierTableInitOpGenericAdaptor::ExperimentalGSanMBarrierTableInitOpGenericAdaptor;
  ExperimentalGSanMBarrierTableInitOpAdaptor(ExperimentalGSanMBarrierTableInitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanMBarrierTableInitOp : public ::mlir::Op<ExperimentalGSanMBarrierTableInitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanMBarrierTableInitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanMBarrierTableInitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_scratch = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("capacity")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCapacityAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCapacityAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_mbarrier_table_init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getScratch() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getScratchMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getCapacityAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().capacity);
  }

  uint32_t getCapacity();
  void setCapacityAttr(::mlir::IntegerAttr attr) {
    getProperties().capacity = attr;
  }

  void setCapacity(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, ::mlir::IntegerAttr capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, ::mlir::IntegerAttr capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, ::mlir::IntegerAttr capacity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::IntegerAttr capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::IntegerAttr capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::IntegerAttr capacity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, uint32_t capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, uint32_t capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, uint32_t capacity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, uint32_t capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, uint32_t capacity);
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, uint32_t capacity);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierTableInitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanMBarrierTableInitOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanMBarrierTableInitOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanMBarrierWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanMBarrierWaitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanMBarrierWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_mbarrier_wait", odsAttrs.getContext());
  }

  ExperimentalGSanMBarrierWaitOpGenericAdaptorBase(ExperimentalGSanMBarrierWaitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanMBarrierWaitOpGenericAdaptor : public detail::ExperimentalGSanMBarrierWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanMBarrierWaitOpGenericAdaptorBase;
  static constexpr int odsIndex_scratch = 0;
  static constexpr int odsIndex_barrier = 1;
  static constexpr int odsIndex_phase = 2;
  static constexpr int odsIndex_pred = 3;
public:
  ExperimentalGSanMBarrierWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanMBarrierWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanMBarrierWaitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanMBarrierWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanMBarrierWaitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanMBarrierWaitOpGenericAdaptor(RangeT values, const ExperimentalGSanMBarrierWaitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanMBarrierWaitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanMBarrierWaitOp>>>
  ExperimentalGSanMBarrierWaitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScratch() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBarrier() {
    return (*getODSOperands(1).begin());
  }

  ValueT getPhase() {
    return (*getODSOperands(2).begin());
  }

  ValueT getPred() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanMBarrierWaitOpAdaptor : public ExperimentalGSanMBarrierWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanMBarrierWaitOpGenericAdaptor::ExperimentalGSanMBarrierWaitOpGenericAdaptor;
  ExperimentalGSanMBarrierWaitOpAdaptor(ExperimentalGSanMBarrierWaitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanMBarrierWaitOp : public ::mlir::Op<ExperimentalGSanMBarrierWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanMBarrierWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanMBarrierWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_scratch = 0;
  static constexpr int odsIndex_barrier = 1;
  static constexpr int odsIndex_phase = 2;
  static constexpr int odsIndex_pred = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_mbarrier_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getScratch() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getBarrier() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPhase() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getScratchMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBarrierMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPhaseMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPredMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value phase, ::mlir::Value pred);
  static ExperimentalGSanMBarrierWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value phase, ::mlir::Value pred);
  static ExperimentalGSanMBarrierWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value phase, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value phase, ::mlir::Value pred);
  static ExperimentalGSanMBarrierWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value phase, ::mlir::Value pred);
  static ExperimentalGSanMBarrierWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value scratch, ::mlir::Value barrier, ::mlir::Value phase, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanMBarrierWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanMBarrierWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanMBarrierWaitOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanTensorAccessOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanTensorAccessOpGenericAdaptorBase {
public:
  struct Properties {
    using isStoreTy = ::mlir::BoolAttr;
    isStoreTy isStore;

    auto getIsStore() const {
      auto &propStorage = this->isStore;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setIsStore(const ::mlir::BoolAttr &propValue) {
      this->isStore = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.isStore == this->isStore &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanTensorAccessOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_tensor_access", odsAttrs.getContext());
  }

  ExperimentalGSanTensorAccessOpGenericAdaptorBase(ExperimentalGSanTensorAccessOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::BoolAttr getIsStoreAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().isStore);
    return attr;
  }

  bool getIsStore();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanTensorAccessOpGenericAdaptor : public detail::ExperimentalGSanTensorAccessOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanTensorAccessOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_mask = 1;
public:
  ExperimentalGSanTensorAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanTensorAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanTensorAccessOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanTensorAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalGSanTensorAccessOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanTensorAccessOpGenericAdaptor(RangeT values, const ExperimentalGSanTensorAccessOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanTensorAccessOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanTensorAccessOp>>>
  ExperimentalGSanTensorAccessOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanTensorAccessOpAdaptor : public ExperimentalGSanTensorAccessOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanTensorAccessOpGenericAdaptor::ExperimentalGSanTensorAccessOpGenericAdaptor;
  ExperimentalGSanTensorAccessOpAdaptor(ExperimentalGSanTensorAccessOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanTensorAccessOp : public ::mlir::Op<ExperimentalGSanTensorAccessOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanTensorAccessOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanTensorAccessOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_mask = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isStore")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsStoreAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsStoreAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_tensor_access");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMask() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::BoolAttr getIsStoreAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().isStore);
  }

  bool getIsStore();
  void setIsStoreAttr(::mlir::BoolAttr attr) {
    getProperties().isStore = attr;
  }

  void setIsStore(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::BoolAttr isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, ::mlir::BoolAttr isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, /*optional*/::mlir::Value mask, bool isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, /*optional*/::mlir::Value mask, bool isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, /*optional*/::mlir::Value mask, bool isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, bool isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, bool isStore);
  static ExperimentalGSanTensorAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, /*optional*/::mlir::Value mask, bool isStore);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanTensorAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanTensorAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanTensorAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanTensorAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanTensorAccessOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanTensorAccessOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalGSanTensorDescAccessOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalGSanTensorDescAccessOpGenericAdaptorBase {
public:
  struct Properties {
    using isStoreTy = ::mlir::BoolAttr;
    isStoreTy isStore;

    auto getIsStore() const {
      auto &propStorage = this->isStore;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setIsStore(const ::mlir::BoolAttr &propValue) {
      this->isStore = propValue;
    }
    using operandSegmentSizesTy = std::array<int32_t, 3>;
    operandSegmentSizesTy operandSegmentSizes;
    ::llvm::ArrayRef<int32_t> getOperandSegmentSizes() const {
      auto &propStorage = this->operandSegmentSizes;
      return propStorage;
    }
    void setOperandSegmentSizes(::llvm::ArrayRef<int32_t> propValue) {
      auto &propStorage = this->operandSegmentSizes;
      ::llvm::copy(propValue, propStorage.begin());
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.isStore == this->isStore &&
        rhs.operandSegmentSizes == this->operandSegmentSizes &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalGSanTensorDescAccessOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_gsan_tensordesc_access", odsAttrs.getContext());
  }

  ExperimentalGSanTensorDescAccessOpGenericAdaptorBase(ExperimentalGSanTensorDescAccessOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::BoolAttr getIsStoreAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().isStore);
    return attr;
  }

  bool getIsStore();
};

} // namespace detail
template <typename RangeT>
class ExperimentalGSanTensorDescAccessOpGenericAdaptor : public detail::ExperimentalGSanTensorDescAccessOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalGSanTensorDescAccessOpGenericAdaptorBase;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_coords = 1;
  static constexpr int odsIndex_pred = 2;
public:
  ExperimentalGSanTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalGSanTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalGSanTensorDescAccessOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalGSanTensorDescAccessOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs) : ExperimentalGSanTensorDescAccessOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalGSanTensorDescAccessOpGenericAdaptor(RangeT values, const ExperimentalGSanTensorDescAccessOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalGSanTensorDescAccessOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalGSanTensorDescAccessOp>>>
  ExperimentalGSanTensorDescAccessOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDesc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getCoords() {
    return getODSOperands(1);
  }

  ValueT getPred() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalGSanTensorDescAccessOpAdaptor : public ExperimentalGSanTensorDescAccessOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalGSanTensorDescAccessOpGenericAdaptor::ExperimentalGSanTensorDescAccessOpGenericAdaptor;
  ExperimentalGSanTensorDescAccessOpAdaptor(ExperimentalGSanTensorDescAccessOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalGSanTensorDescAccessOp : public ::mlir::Op<ExperimentalGSanTensorDescAccessOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalGSanTensorDescAccessOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalGSanTensorDescAccessOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_coords = 1;
  static constexpr int odsIndex_pred = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isStore"), ::llvm::StringRef("operandSegmentSizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsStoreAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsStoreAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
   return (*this)->getName().getAttributeNames().back();
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
   return name.getAttributeNames().back();
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_gsan_tensordesc_access");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::TensorDescInterface> getDesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::TensorDescInterface>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getCoords() {
    return getODSOperands(1);
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getDescMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getCoordsMutable();
  ::mlir::MutableOperandRange getPredMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::BoolAttr getIsStoreAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().isStore);
  }

  bool getIsStore();
  void setIsStoreAttr(::mlir::BoolAttr attr) {
    getProperties().isStore = attr;
  }

  void setIsStore(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, ::mlir::BoolAttr isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, bool isStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, bool isStore);
  static ExperimentalGSanTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange coords, /*optional*/::mlir::Value pred, bool isStore);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalGSanTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanTensorDescAccessOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalGSanTensorDescAccessOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalGSanTensorDescAccessOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalGSanTensorDescAccessOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLocalGatherOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalLocalGatherOpGenericAdaptorBase {
public:
  struct Properties {
    using axisTy = ::mlir::IntegerAttr;
    axisTy axis;

    auto getAxis() const {
      auto &propStorage = this->axis;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setAxis(const ::mlir::IntegerAttr &propValue) {
      this->axis = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.axis == this->axis &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalLocalGatherOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_local_gather", odsAttrs.getContext());
  }

  ExperimentalLocalGatherOpGenericAdaptorBase(ExperimentalLocalGatherOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().axis);
    return attr;
  }

  uint32_t getAxis();
};

} // namespace detail
template <typename RangeT>
class ExperimentalLocalGatherOpGenericAdaptor : public detail::ExperimentalLocalGatherOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalLocalGatherOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_indices = 1;
  static constexpr int odsIndex_offsets = 2;
public:
  ExperimentalLocalGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalLocalGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalLocalGatherOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalLocalGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalLocalGatherOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalLocalGatherOpGenericAdaptor(RangeT values, const ExperimentalLocalGatherOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalLocalGatherOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalLocalGatherOp>>>
  ExperimentalLocalGatherOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIndices() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOffsets() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalLocalGatherOpAdaptor : public ExperimentalLocalGatherOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalLocalGatherOpGenericAdaptor::ExperimentalLocalGatherOpGenericAdaptor;
  ExperimentalLocalGatherOpAdaptor(ExperimentalLocalGatherOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalLocalGatherOp : public ::mlir::Op<ExperimentalLocalGatherOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalLocalGatherOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalLocalGatherOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_indices = 1;
  static constexpr int odsIndex_offsets = 2;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_local_gather");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getIndices() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getOffsets() {
    return getODSOperands(2);
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIndicesMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getOffsetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().axis);
  }

  uint32_t getAxis();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    getProperties().axis = attr;
  }

  void setAxis(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, ::mlir::IntegerAttr axis);
  static ExperimentalLocalGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, ::mlir::IntegerAttr axis);
  static ExperimentalLocalGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, ::mlir::IntegerAttr axis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, ::mlir::IntegerAttr axis);
  static ExperimentalLocalGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, ::mlir::IntegerAttr axis);
  static ExperimentalLocalGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, ::mlir::IntegerAttr axis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, uint32_t axis);
  static ExperimentalLocalGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, uint32_t axis);
  static ExperimentalLocalGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, uint32_t axis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, uint32_t axis);
  static ExperimentalLocalGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, uint32_t axis);
  static ExperimentalLocalGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value indices, ::mlir::ValueRange offsets, uint32_t axis);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLocalGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLocalGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLocalGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLocalGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLocalGatherOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalLocalGatherOpGenericAdaptorBase::Properties)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLockAcquireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalLockAcquireOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalLockAcquireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_lock_acquire", odsAttrs.getContext());
  }

  ExperimentalLockAcquireOpGenericAdaptorBase(ExperimentalLockAcquireOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalLockAcquireOpGenericAdaptor : public detail::ExperimentalLockAcquireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalLockAcquireOpGenericAdaptorBase;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
public:
  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalLockAcquireOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalLockAcquireOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, const ExperimentalLockAcquireOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalLockAcquireOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalLockAcquireOp>>>
  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLock() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalLockAcquireOpAdaptor : public ExperimentalLockAcquireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalLockAcquireOpGenericAdaptor::ExperimentalLockAcquireOpGenericAdaptor;
  ExperimentalLockAcquireOpAdaptor(ExperimentalLockAcquireOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalLockAcquireOp : public ::mlir::Op<ExperimentalLockAcquireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalLockAcquireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalLockAcquireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_lock_acquire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getLock() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getLockMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getPredMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLockAcquireOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLockReleaseOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalLockReleaseOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalLockReleaseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_lock_release", odsAttrs.getContext());
  }

  ExperimentalLockReleaseOpGenericAdaptorBase(ExperimentalLockReleaseOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalLockReleaseOpGenericAdaptor : public detail::ExperimentalLockReleaseOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalLockReleaseOpGenericAdaptorBase;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
public:
  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalLockReleaseOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalLockReleaseOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, const ExperimentalLockReleaseOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalLockReleaseOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalLockReleaseOp>>>
  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLock() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalLockReleaseOpAdaptor : public ExperimentalLockReleaseOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalLockReleaseOpGenericAdaptor::ExperimentalLockReleaseOpGenericAdaptor;
  ExperimentalLockReleaseOpAdaptor(ExperimentalLockReleaseOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalLockReleaseOp : public ::mlir::Op<ExperimentalLockReleaseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalLockReleaseOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalLockReleaseOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_lock_release");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getLock() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getLockMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getPredMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLockReleaseOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalMemDescToI32Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalMemDescToI32OpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalMemDescToI32OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_memdesc_to_i32", odsAttrs.getContext());
  }

  ExperimentalMemDescToI32OpGenericAdaptorBase(ExperimentalMemDescToI32Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ExperimentalMemDescToI32OpGenericAdaptor : public detail::ExperimentalMemDescToI32OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalMemDescToI32OpGenericAdaptorBase;
  static constexpr int odsIndex_memdesc = 0;
public:
  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalMemDescToI32OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalMemDescToI32OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, const ExperimentalMemDescToI32OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalMemDescToI32Op, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalMemDescToI32Op>>>
  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemdesc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalMemDescToI32OpAdaptor : public ExperimentalMemDescToI32OpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalMemDescToI32OpGenericAdaptor::ExperimentalMemDescToI32OpGenericAdaptor;
  ExperimentalMemDescToI32OpAdaptor(ExperimentalMemDescToI32Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalMemDescToI32Op : public ::mlir::Op<ExperimentalMemDescToI32Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalMemDescToI32OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalMemDescToI32OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_memdesc = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_memdesc_to_i32");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getMemdesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getMemdescMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value memdesc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value memdesc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::PropertyRef properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalMemDescToI32Op)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalMemoryOffsetToI32Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalMemoryOffsetToI32OpGenericAdaptorBase {
public:
  struct Properties {
    using memTypeTy = ::mlir::triton::instrument::MemTypeAttr;
    memTypeTy memType;

    auto getMemType() const {
      auto &propStorage = this->memType;
      return ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(propStorage);
    }
    void setMemType(const ::mlir::triton::instrument::MemTypeAttr &propValue) {
      this->memType = propValue;
    }
    using offsetTy = ::mlir::IntegerAttr;
    offsetTy offset;

    auto getOffset() const {
      auto &propStorage = this->offset;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setOffset(const ::mlir::IntegerAttr &propValue) {
      this->offset = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.memType == this->memType &&
        rhs.offset == this->offset &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalMemoryOffsetToI32OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_memory_offset_to_i32", odsAttrs.getContext());
  }

  ExperimentalMemoryOffsetToI32OpGenericAdaptorBase(ExperimentalMemoryOffsetToI32Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getOffsetAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().offset);
    return attr;
  }

  uint32_t getOffset();
  ::mlir::triton::instrument::MemTypeAttr getMemTypeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(getProperties().memType);
    return attr;
  }

  ::mlir::triton::instrument::MemType getMemType();
};

} // namespace detail
template <typename RangeT>
class ExperimentalMemoryOffsetToI32OpGenericAdaptor : public detail::ExperimentalMemoryOffsetToI32OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalMemoryOffsetToI32OpGenericAdaptorBase;
public:
  ExperimentalMemoryOffsetToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalMemoryOffsetToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ExperimentalMemoryOffsetToI32OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalMemoryOffsetToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalMemoryOffsetToI32OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalMemoryOffsetToI32OpGenericAdaptor(RangeT values, const ExperimentalMemoryOffsetToI32OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalMemoryOffsetToI32Op, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalMemoryOffsetToI32Op>>>
  ExperimentalMemoryOffsetToI32OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalMemoryOffsetToI32OpAdaptor : public ExperimentalMemoryOffsetToI32OpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalMemoryOffsetToI32OpGenericAdaptor::ExperimentalMemoryOffsetToI32OpGenericAdaptor;
  ExperimentalMemoryOffsetToI32OpAdaptor(ExperimentalMemoryOffsetToI32Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalMemoryOffsetToI32Op : public ::mlir::Op<ExperimentalMemoryOffsetToI32Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalMemoryOffsetToI32OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalMemoryOffsetToI32OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("memType"), ::llvm::StringRef("offset")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMemTypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMemTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_memory_offset_to_i32");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getOffsetAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().offset);
  }

  uint32_t getOffset();
  ::mlir::triton::instrument::MemTypeAttr getMemTypeAttr() {
    return ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(getProperties().memType);
  }

  ::mlir::triton::instrument::MemType getMemType();
  void setOffsetAttr(::mlir::IntegerAttr attr) {
    getProperties().offset = attr;
  }

  void setOffset(uint32_t attrValue);
  void setMemTypeAttr(::mlir::triton::instrument::MemTypeAttr attr) {
    getProperties().memType = attr;
  }

  void setMemType(::mlir::triton::instrument::MemType attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr offset, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t offset, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemoryOffsetToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::PropertyRef properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalMemoryOffsetToI32Op)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::detail::ExperimentalMemoryOffsetToI32OpGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

