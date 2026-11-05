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
namespace mlir::triton {

/// Target-neutral cache hints used by Triton memory operations.
/// 
/// Target dialects may define richer cache policy attributes. This
/// attribute represents the cache modifier and eviction policy exposed by
/// the common Triton language API. Cache policies are performance hints and
/// have no memory-ordering semantics.
class CachePolicyAttr;
namespace detail {

struct CachePolicyAttrStorage;

} // namespace detail
class CachePolicyAttr : public ::mlir::Attribute::AttrBase<CachePolicyAttr, ::mlir::Attribute, detail::CachePolicyAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tt.cache_policy";
  static constexpr ::llvm::StringLiteral dialectName = "tt";
  static CachePolicyAttr get(::mlir::MLIRContext *context, ::mlir::triton::CacheModifier cache_modifier, ::mlir::triton::EvictionPolicy eviction_policy);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cache_policy"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::CacheModifier getCacheModifier() const;
  ::mlir::triton::EvictionPolicy getEvictionPolicy() const;
};

} // namespace mlir::triton
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::CachePolicyAttr)

#endif // GET_ATTRDEF_CLASSES

