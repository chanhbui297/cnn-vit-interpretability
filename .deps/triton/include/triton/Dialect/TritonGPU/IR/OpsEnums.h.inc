/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonGPUEnums.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::gpu {

// 
enum class AddrSpace : uint32_t {
  None = 0,
  Local = 1,
  GlobalRead = 2,
  GlobalWrite = 4,
  TensorRead = 8,
  TensorWrite = 16,
  All = 31,
};

::std::optional<AddrSpace> symbolizeAddrSpace(uint32_t);
std::string stringifyAddrSpace(AddrSpace);
::std::optional<AddrSpace> symbolizeAddrSpace(::llvm::StringRef);

inline constexpr AddrSpace operator|(AddrSpace a, AddrSpace b) {
  return static_cast<AddrSpace>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr AddrSpace operator&(AddrSpace a, AddrSpace b) {
  return static_cast<AddrSpace>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr AddrSpace operator^(AddrSpace a, AddrSpace b) {
  return static_cast<AddrSpace>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr AddrSpace &operator|=(AddrSpace &a, AddrSpace b) {
    return a = a | b;
}
inline constexpr AddrSpace &operator&=(AddrSpace &a, AddrSpace b) {
    return a = a & b;
}
inline constexpr AddrSpace &operator^=(AddrSpace &a, AddrSpace b) {
    return a = a ^ b;
}
inline constexpr AddrSpace operator~(AddrSpace bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<AddrSpace>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(31u));
}
inline constexpr bool bitEnumContainsAll(AddrSpace bits, AddrSpace bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(AddrSpace bits, AddrSpace bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr AddrSpace bitEnumClear(AddrSpace bits, AddrSpace bit) {
  return bits & ~bit;
}
inline constexpr AddrSpace bitEnumSet(AddrSpace bits, AddrSpace bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(AddrSpace enumValue) {
  return stringifyAddrSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AddrSpace> symbolizeEnum<AddrSpace>(::llvm::StringRef str) {
  return symbolizeAddrSpace(str);
}

class AddrSpaceAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AddrSpace;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AddrSpaceAttr get(::mlir::MLIRContext *context, AddrSpace val);
  AddrSpace getValue() const;
};

} // namespace mlir::triton::gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::gpu::AddrSpace, ::mlir::triton::gpu::AddrSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::gpu::AddrSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::gpu::AddrSpace> attr = ::mlir::triton::gpu::symbolizeEnum<::mlir::triton::gpu::AddrSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, local, global_read, global_write, tensor_read, tensor_write, all] for , got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::gpu::AddrSpace>, std::optional<::mlir::triton::gpu::AddrSpace>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::gpu::AddrSpace>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"none", "local", "global_read", "global_write", "tensor_read", "tensor_write", "all"})))
      return std::optional<::mlir::triton::gpu::AddrSpace>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::gpu::AddrSpace> attr = ::mlir::triton::gpu::symbolizeEnum<::mlir::triton::gpu::AddrSpace>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, local, global_read, global_write, tensor_read, tensor_write, all] for , got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::gpu::AddrSpace value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::triton::gpu::AddrSpace::All:
    return p << valueStr;
  default:
    break;
  }
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::triton::gpu::AddrSpace>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::gpu::AddrSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::gpu::AddrSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::gpu::AddrSpace &lhs, const ::mlir::triton::gpu::AddrSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::gpu {

// shared-memory access kind
enum class SharedKind : uint32_t {
  Generic = 0,
  Async = 1,
  Barrier = 2,
};

::std::optional<SharedKind> symbolizeSharedKind(uint32_t);
::llvm::StringRef stringifySharedKind(SharedKind);
::std::optional<SharedKind> symbolizeSharedKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSharedKind() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(SharedKind enumValue) {
  return stringifySharedKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SharedKind> symbolizeEnum<SharedKind>(::llvm::StringRef str) {
  return symbolizeSharedKind(str);
}

class SharedKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = SharedKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static SharedKindAttr get(::mlir::MLIRContext *context, SharedKind val);
  SharedKind getValue() const;
};

} // namespace mlir::triton::gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::gpu::SharedKind, ::mlir::triton::gpu::SharedKind> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::gpu::SharedKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for shared-memory access kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::gpu::SharedKind> attr = ::mlir::triton::gpu::symbolizeEnum<::mlir::triton::gpu::SharedKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [generic, async, barrier] for shared-memory access kind, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::gpu::SharedKind>, std::optional<::mlir::triton::gpu::SharedKind>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::gpu::SharedKind>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"generic", "async", "barrier"})))
      return std::optional<::mlir::triton::gpu::SharedKind>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::gpu::SharedKind> attr = ::mlir::triton::gpu::symbolizeEnum<::mlir::triton::gpu::SharedKind>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [generic, async, barrier] for shared-memory access kind, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::gpu::SharedKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::gpu::SharedKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::gpu::SharedKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::gpu::SharedKind &lhs, const ::mlir::triton::gpu::SharedKind &rhs) {
    return lhs == rhs;
  }
};
}

