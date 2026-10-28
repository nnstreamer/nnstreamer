/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        script.h
 * @date        27 Oct 2026
 * @brief       A reduced stand-in of the LibTorch C++ API for testing
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * This header replaces <torch/script.h> when tensor_filter_pytorch.cc is
 * compiled for the mock-based unit test. It declares only what the sub-plugin
 * uses, written from the public LibTorch API reference; it is neither a copy
 * of nor ABI-compatible with LibTorch. Tensors live in host memory only, and
 * the behaviour of a "model" is chosen by the content of the file that
 * torch::jit::load() reads (see torch_mock.h).
 */
#ifndef __NNS_TORCH_MOCK_SCRIPT_H__
#define __NNS_TORCH_MOCK_SCRIPT_H__

#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace c10
{
/**
 * @brief Non-owning view of a contiguous array.
 */
template <typename T> class ArrayRef
{
  public:
  /** @brief Construct an empty view. */
  ArrayRef () : data_ (nullptr), length_ (0)
  {
  }
  /** @brief View the elements of a vector. */
  ArrayRef (const std::vector<T> &vec)
      : data_ (vec.data ()), length_ (vec.size ())
  {
  }
  /** @brief The first element. */
  const T *begin () const
  {
    return data_;
  }
  /** @brief One past the last element. */
  const T *end () const
  {
    return data_ + length_;
  }
  /** @brief The number of elements. */
  size_t size () const
  {
    return length_;
  }
  /**
   * @brief The idx-th element.
   * @details LibTorch leaves the bound to the caller; the mock throws instead,
   * so that a test sees an out-of-bound read.
   */
  const T &operator[] (size_t idx) const
  {
    if (idx >= length_)
      throw std::out_of_range ("mock ArrayRef index out of range");
    return data_[idx];
  }

  private:
  const T *data_;
  size_t length_;
};

using IntArrayRef = ArrayRef<int64_t>;

/**
 * @brief Element types of a tensor.
 */
enum class ScalarType : int8_t {
  Byte,
  Char,
  Short,
  Int,
  Long,
  Half,
  Float,
  Double,
  Bool,
};

/**
 * @brief Devices a tensor or a module can live on.
 */
enum class DeviceType : int8_t {
  CPU,
  CUDA,
};

/**
 * @brief A device a model is loaded onto.
 */
class Device
{
  public:
  /** @brief Construct a device of the given type. */
  Device (DeviceType type) : type_ (type)
  {
  }
  /** @brief The type of this device. */
  DeviceType type () const
  {
    return type_;
  }

  private:
  DeviceType type_;
};

constexpr ScalarType kByte = ScalarType::Byte;
constexpr ScalarType kChar = ScalarType::Char;
constexpr ScalarType kShort = ScalarType::Short;
constexpr ScalarType kInt = ScalarType::Int;
constexpr ScalarType kLong = ScalarType::Long;
constexpr ScalarType kHalf = ScalarType::Half;
constexpr ScalarType kFloat = ScalarType::Float;
constexpr ScalarType kDouble = ScalarType::Double;
constexpr ScalarType kBool = ScalarType::Bool;
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;

/**
 * @brief The byte size of an element of the given type.
 */
inline size_t
elementSize (ScalarType type)
{
  switch (type) {
    case ScalarType::Byte:
    case ScalarType::Char:
    case ScalarType::Bool:
      return 1;
    case ScalarType::Short:
    case ScalarType::Half:
      return 2;
    case ScalarType::Int:
    case ScalarType::Float:
      return 4;
    default:
      return 8;
  }
}
} /* namespace c10 */

namespace at
{
using c10::Device;
using c10::DeviceType;
using c10::IntArrayRef;
using c10::kCPU;
using c10::kCUDA;
using c10::ScalarType;

/**
 * @brief Options a tensor is created with.
 */
class TensorOptions
{
  public:
  /** @brief Default options: float32. */
  TensorOptions () : dtype_ (ScalarType::Float)
  {
  }
  /** @brief Return a copy of the options with the given element type. */
  TensorOptions dtype (ScalarType type) const
  {
    TensorOptions options (*this);
    options.dtype_ = type;
    return options;
  }
  /** @brief The element type of the options. */
  ScalarType dtype () const
  {
    return dtype_;
  }

  private:
  ScalarType dtype_;
};

/**
 * @brief A dense tensor in host memory.
 * @details The data are either borrowed (from_blob) or owned and shared
 * between copies, like the storage of a LibTorch tensor.
 */
class Tensor
{
  public:
  /** @brief Construct an undefined tensor. */
  Tensor ()
      : type_ (ScalarType::Float), device_ (DeviceType::CPU), data_ (nullptr)
  {
  }

  /** @brief Construct a tensor of the given shape that owns zeroed data. */
  Tensor (const std::vector<int64_t> &sizes, ScalarType type)
      : sizes_ (sizes), type_ (type), device_ (DeviceType::CPU)
  {
    owned_ = std::make_shared<std::vector<uint8_t>> (nbytes ());
    data_ = owned_->data ();
  }

  /** @brief Construct a tensor that borrows the given data. */
  Tensor (void *data, const std::vector<int64_t> &sizes, ScalarType type)
      : sizes_ (sizes), type_ (type), device_ (DeviceType::CPU), data_ (data)
  {
  }

  /** @brief The size of each dimension. */
  IntArrayRef sizes () const
  {
    return IntArrayRef (sizes_);
  }
  /** @brief The number of dimensions. */
  int64_t ndimension () const
  {
    return static_cast<int64_t> (sizes_.size ());
  }
  /** @brief The element type. */
  ScalarType scalar_type () const
  {
    return type_;
  }
  /** @brief The device the tensor lives on. */
  DeviceType device_type () const
  {
    return device_;
  }
  /** @brief The number of elements. */
  int64_t numel () const
  {
    int64_t n = 1;
    for (int64_t s : sizes_)
      n *= s;
    return n;
  }
  /** @brief The byte size of the data. */
  size_t nbytes () const
  {
    return static_cast<size_t> (numel ()) * c10::elementSize (type_);
  }
  /** @brief The data. */
  void *data_ptr () const
  {
    return data_;
  }
  /**
   * @brief A view of the whole tensor along the given dimension.
   * @details Like LibTorch, a 0-dim tensor cannot be sliced.
   */
  Tensor slice (int64_t dim = 0) const
  {
    (void) dim;
    if (sizes_.empty ())
      throw std::out_of_range ("slice() cannot be applied to a 0-dim tensor.");
    return *this;
  }
  /** @brief The tensor itself; the mock keeps every tensor contiguous. */
  Tensor contiguous () const
  {
    return *this;
  }
  /** @brief The same data, tagged as living on the given device. */
  Tensor to (DeviceType device) const
  {
    Tensor tensor (*this);
    tensor.device_ = device;
    return tensor;
  }

  private:
  std::vector<int64_t> sizes_;
  ScalarType type_;
  DeviceType device_;
  void *data_;
  std::shared_ptr<std::vector<uint8_t>> owned_;
};
} /* namespace at */

namespace c10
{
namespace ivalue
{
struct Tuple;
}

/**
 * @brief A value a TorchScript module takes or returns.
 */
class IValue
{
  public:
  /** @brief The kinds of value the mock knows. */
  enum class Tag { None, Tensor, Tuple, List };

  /** @brief Construct a None value. */
  IValue () : tag_ (Tag::None)
  {
  }
  /** @brief Construct a tensor value. */
  IValue (const at::Tensor &tensor) : tag_ (Tag::Tensor), tensor_ (tensor)
  {
  }
  /** @brief Construct a tuple value. */
  IValue (std::shared_ptr<ivalue::Tuple> tuple)
      : tag_ (Tag::Tuple), tuple_ (tuple)
  {
  }
  /** @brief Construct a list value. */
  static IValue makeList (const std::vector<IValue> &elements)
  {
    IValue value;
    value.tag_ = Tag::List;
    value.list_ = std::make_shared<std::vector<IValue>> (elements);
    return value;
  }

  /** @brief Whether the value is a tensor. */
  bool isTensor () const
  {
    return tag_ == Tag::Tensor;
  }
  /** @brief Whether the value is a tuple. */
  bool isTuple () const
  {
    return tag_ == Tag::Tuple;
  }
  /** @brief Whether the value is a list. */
  bool isList () const
  {
    return tag_ == Tag::List;
  }
  /** @brief The tensor of a tensor value. */
  at::Tensor toTensor () const
  {
    if (tag_ != Tag::Tensor)
      throw std::runtime_error ("Expected Tensor");
    return tensor_;
  }
  /** @brief The tuple of a tuple value. */
  std::shared_ptr<ivalue::Tuple> toTuple () const
  {
    if (tag_ != Tag::Tuple)
      throw std::runtime_error ("Expected Tuple");
    return tuple_;
  }
  /** @brief The elements of a list value. */
  ArrayRef<IValue> toListRef () const
  {
    if (tag_ != Tag::List)
      throw std::runtime_error ("Expected GenericList");
    return ArrayRef<IValue> (*list_);
  }

  private:
  Tag tag_;
  at::Tensor tensor_;
  std::shared_ptr<ivalue::Tuple> tuple_;
  std::shared_ptr<std::vector<IValue>> list_;
};

namespace ivalue
{
/**
 * @brief The elements of a tuple value.
 */
struct Tuple {
  std::vector<IValue> elements_; /**< the elements */

  /** @brief The elements of the tuple. */
  const std::vector<IValue> &elements () const
  {
    return elements_;
  }
};
} /* namespace ivalue */
} /* namespace c10 */

namespace torch
{
using at::Tensor;
using at::TensorOptions;
using c10::Device;
using c10::IValue;
using c10::kCPU;
using c10::kCUDA;
using Dtype = c10::ScalarType;

constexpr Dtype kU8 = c10::kByte;
constexpr Dtype kI8 = c10::kChar;
constexpr Dtype kI16 = c10::kShort;
constexpr Dtype kI32 = c10::kInt;
constexpr Dtype kI64 = c10::kLong;
constexpr Dtype kF16 = c10::kHalf;
constexpr Dtype kF32 = c10::kFloat;
constexpr Dtype kF64 = c10::kDouble;

/**
 * @brief Wrap the given data in a tensor without copying them.
 */
inline at::Tensor
from_blob (void *data, const std::vector<int64_t> &sizes, const TensorOptions &options)
{
  return at::Tensor (data, sizes, options.dtype ());
}

namespace autograd
{
using Variable = at::Tensor;
}

namespace jit
{
using c10::IValue;

/**
 * @brief A loaded TorchScript module.
 */
class Module
{
  public:
  /** @brief Construct a module of the given mock behaviour. */
  explicit Module (const std::string &behaviour) : behaviour_ (behaviour)
  {
  }
  /** @brief Switch the module to evaluation mode. */
  void eval ();
  /** @brief Run the module. */
  IValue forward (std::vector<IValue> inputs);

  private:
  std::string behaviour_;
};

namespace script
{
using Module = torch::jit::Module;
}

/**
 * @brief Load a TorchScript module from the given file onto the given device.
 */
Module load (const std::string &filename, c10::Device device);
} /* namespace jit */
} /* namespace torch */

#endif /* __NNS_TORCH_MOCK_SCRIPT_H__ */
