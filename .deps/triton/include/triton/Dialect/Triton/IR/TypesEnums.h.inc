/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonTypeEnums.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton {

// allowed 32-bit signless integer cases: 1, 0, 4
enum class PtrAddrSpace : uint32_t {
  Global = 1,
  Descriptor = 0,
  Constant = 4,
};

::std::optional<PtrAddrSpace> symbolizePtrAddrSpace(uint32_t);
::llvm::StringRef stringifyPtrAddrSpace(PtrAddrSpace);
::std::optional<PtrAddrSpace> symbolizePtrAddrSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPtrAddrSpace() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(PtrAddrSpace enumValue) {
  return stringifyPtrAddrSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PtrAddrSpace> symbolizeEnum<PtrAddrSpace>(::llvm::StringRef str) {
  return symbolizePtrAddrSpace(str);
}

class PtrAddrSpaceAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = PtrAddrSpace;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static PtrAddrSpaceAttr get(::mlir::MLIRContext *context, PtrAddrSpace val);
  PtrAddrSpace getValue() const;
};

} // namespace mlir::triton

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::PtrAddrSpace, ::mlir::triton::PtrAddrSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::PtrAddrSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 1, 0, 4");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::PtrAddrSpace> attr = ::mlir::triton::symbolizeEnum<::mlir::triton::PtrAddrSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [global, descriptor, constant] for allowed 32-bit signless integer cases: 1, 0, 4, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::PtrAddrSpace>, std::optional<::mlir::triton::PtrAddrSpace>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::PtrAddrSpace>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"global", "descriptor", "constant"})))
      return std::optional<::mlir::triton::PtrAddrSpace>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::PtrAddrSpace> attr = ::mlir::triton::symbolizeEnum<::mlir::triton::PtrAddrSpace>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [global, descriptor, constant] for allowed 32-bit signless integer cases: 1, 0, 4, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::PtrAddrSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::PtrAddrSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::PtrAddrSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::PtrAddrSpace &lhs, const ::mlir::triton::PtrAddrSpace &rhs) {
    return lhs == rhs;
  }
};
}

