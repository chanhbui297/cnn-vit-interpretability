/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonNvidiaGPUEnums.td                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvidia_gpu {

// NVIDIA cache eviction priority
enum class CacheEvictionPriority : uint32_t {
  NONE = 0,
  EVICT_NORMAL = 1,
  EVICT_UNCHANGED = 2,
  EVICT_FIRST = 3,
  EVICT_LAST = 4,
  NO_ALLOCATE = 5,
};

::std::optional<CacheEvictionPriority> symbolizeCacheEvictionPriority(uint32_t);
::llvm::StringRef stringifyCacheEvictionPriority(CacheEvictionPriority);
::std::optional<CacheEvictionPriority> symbolizeCacheEvictionPriority(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCacheEvictionPriority() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(CacheEvictionPriority enumValue) {
  return stringifyCacheEvictionPriority(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CacheEvictionPriority> symbolizeEnum<CacheEvictionPriority>(::llvm::StringRef str) {
  return symbolizeCacheEvictionPriority(str);
}

} // namespace mlir::triton::nvidia_gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvidia_gpu::CacheEvictionPriority, ::mlir::triton::nvidia_gpu::CacheEvictionPriority> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvidia_gpu::CacheEvictionPriority> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVIDIA cache eviction priority");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::CacheEvictionPriority> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::CacheEvictionPriority>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, evict_normal, evict_unchanged, evict_first, evict_last, no_allocate] for NVIDIA cache eviction priority, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvidia_gpu::CacheEvictionPriority>, std::optional<::mlir::triton::nvidia_gpu::CacheEvictionPriority>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvidia_gpu::CacheEvictionPriority>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"none", "evict_normal", "evict_unchanged", "evict_first", "evict_last", "no_allocate"})))
      return std::optional<::mlir::triton::nvidia_gpu::CacheEvictionPriority>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::CacheEvictionPriority> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::CacheEvictionPriority>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, evict_normal, evict_unchanged, evict_first, evict_last, no_allocate] for NVIDIA cache eviction priority, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvidia_gpu::CacheEvictionPriority value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvidia_gpu::CacheEvictionPriority> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::nvidia_gpu::CacheEvictionPriority &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvidia_gpu::CacheEvictionPriority &lhs, const ::mlir::triton::nvidia_gpu::CacheEvictionPriority &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::nvidia_gpu {

// packed arithmetic operation
enum class PackedArithOpKind : uint32_t {
  ADD = 0,
  SUB = 1,
  MUL = 2,
  FMA = 3,
  MIN = 4,
  MAX = 5,
};

::std::optional<PackedArithOpKind> symbolizePackedArithOpKind(uint32_t);
::llvm::StringRef stringifyPackedArithOpKind(PackedArithOpKind);
::std::optional<PackedArithOpKind> symbolizePackedArithOpKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPackedArithOpKind() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(PackedArithOpKind enumValue) {
  return stringifyPackedArithOpKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PackedArithOpKind> symbolizeEnum<PackedArithOpKind>(::llvm::StringRef str) {
  return symbolizePackedArithOpKind(str);
}

class PackedArithOpKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = PackedArithOpKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static PackedArithOpKindAttr get(::mlir::MLIRContext *context, PackedArithOpKind val);
  PackedArithOpKind getValue() const;
};

} // namespace mlir::triton::nvidia_gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvidia_gpu::PackedArithOpKind, ::mlir::triton::nvidia_gpu::PackedArithOpKind> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvidia_gpu::PackedArithOpKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for packed arithmetic operation");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::PackedArithOpKind> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::PackedArithOpKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [add, sub, mul, fma, min, max] for packed arithmetic operation, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvidia_gpu::PackedArithOpKind>, std::optional<::mlir::triton::nvidia_gpu::PackedArithOpKind>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvidia_gpu::PackedArithOpKind>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"add", "sub", "mul", "fma", "min", "max"})))
      return std::optional<::mlir::triton::nvidia_gpu::PackedArithOpKind>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::PackedArithOpKind> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::PackedArithOpKind>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [add, sub, mul, fma, min, max] for packed arithmetic operation, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvidia_gpu::PackedArithOpKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvidia_gpu::PackedArithOpKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::nvidia_gpu::PackedArithOpKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvidia_gpu::PackedArithOpKind &lhs, const ::mlir::triton::nvidia_gpu::PackedArithOpKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::nvidia_gpu {

// allowed 32-bit signless integer cases: 1, 2
enum class TMEMLoadReduceModifier : uint32_t {
  MIN = 1,
  MAX = 2,
};

::std::optional<TMEMLoadReduceModifier> symbolizeTMEMLoadReduceModifier(uint32_t);
::llvm::StringRef stringifyTMEMLoadReduceModifier(TMEMLoadReduceModifier);
::std::optional<TMEMLoadReduceModifier> symbolizeTMEMLoadReduceModifier(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTMEMLoadReduceModifier() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(TMEMLoadReduceModifier enumValue) {
  return stringifyTMEMLoadReduceModifier(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TMEMLoadReduceModifier> symbolizeEnum<TMEMLoadReduceModifier>(::llvm::StringRef str) {
  return symbolizeTMEMLoadReduceModifier(str);
}

} // namespace mlir::triton::nvidia_gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier, ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 1, 2");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [min, max] for allowed 32-bit signless integer cases: 1, 2, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>, std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"min", "max"})))
      return std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [min, max] for allowed 32-bit signless integer cases: 1, 2, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier &lhs, const ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::nvidia_gpu {

// tensor memory wait kind
enum class TMEMWaitKind : uint32_t {
  LOAD = 0,
  STORE = 1,
};

::std::optional<TMEMWaitKind> symbolizeTMEMWaitKind(uint32_t);
::llvm::StringRef stringifyTMEMWaitKind(TMEMWaitKind);
::std::optional<TMEMWaitKind> symbolizeTMEMWaitKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTMEMWaitKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(TMEMWaitKind enumValue) {
  return stringifyTMEMWaitKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TMEMWaitKind> symbolizeEnum<TMEMWaitKind>(::llvm::StringRef str) {
  return symbolizeTMEMWaitKind(str);
}

class TMEMWaitKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = TMEMWaitKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static TMEMWaitKindAttr get(::mlir::MLIRContext *context, TMEMWaitKind val);
  TMEMWaitKind getValue() const;
};

} // namespace mlir::triton::nvidia_gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvidia_gpu::TMEMWaitKind, ::mlir::triton::nvidia_gpu::TMEMWaitKind> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvidia_gpu::TMEMWaitKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for tensor memory wait kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TMEMWaitKind> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TMEMWaitKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [load, store] for tensor memory wait kind, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvidia_gpu::TMEMWaitKind>, std::optional<::mlir::triton::nvidia_gpu::TMEMWaitKind>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvidia_gpu::TMEMWaitKind>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"load", "store"})))
      return std::optional<::mlir::triton::nvidia_gpu::TMEMWaitKind>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TMEMWaitKind> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TMEMWaitKind>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [load, store] for tensor memory wait kind, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvidia_gpu::TMEMWaitKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvidia_gpu::TMEMWaitKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::nvidia_gpu::TMEMWaitKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvidia_gpu::TMEMWaitKind &lhs, const ::mlir::triton::nvidia_gpu::TMEMWaitKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::nvidia_gpu {

// allowed 32-bit signless integer cases: 0, 1
enum class TensorMemoryScalesBlockRepOrder : uint32_t {
  MN_THEN_K = 0,
  K_THEN_MN = 1,
};

::std::optional<TensorMemoryScalesBlockRepOrder> symbolizeTensorMemoryScalesBlockRepOrder(uint32_t);
::llvm::StringRef stringifyTensorMemoryScalesBlockRepOrder(TensorMemoryScalesBlockRepOrder);
::std::optional<TensorMemoryScalesBlockRepOrder> symbolizeTensorMemoryScalesBlockRepOrder(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTensorMemoryScalesBlockRepOrder() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(TensorMemoryScalesBlockRepOrder enumValue) {
  return stringifyTensorMemoryScalesBlockRepOrder(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TensorMemoryScalesBlockRepOrder> symbolizeEnum<TensorMemoryScalesBlockRepOrder>(::llvm::StringRef str) {
  return symbolizeTensorMemoryScalesBlockRepOrder(str);
}

} // namespace mlir::triton::nvidia_gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [mnThenK, kThenMn] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder>, std::optional<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"mnThenK", "kThenMn"})))
      return std::optional<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [mnThenK, kThenMn] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder &lhs, const ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder &rhs) {
    return lhs == rhs;
  }
};
}

