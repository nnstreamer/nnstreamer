/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    snpe_mock_api_v1.hpp
 * @date    22 Sep 2026
 * @brief   Mock of the SNPE 1.x C++ API used by tensor_filter_snpe_v1.cc.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * Written from the vendor's public C++ API reference of the Qualcomm Neural
 * Processing SDK (https://docs.qualcomm.com/); no header of the proprietary
 * SDK is copied here. Only the members that
 * ext/nnstreamer/tensor_filter/tensor_filter_snpe_v1.cc refers to are declared.
 *
 * Deviations from the real SDK, all deliberate, because a test binary uses
 * these headers and the mock library as one pair and ABI compatibility is not
 * a goal:
 * - ITensor is a concrete class over a float vector and its iterators are raw
 *   float pointers, where the SDK exposes an abstract class and its own
 *   iterator type.
 * - Only the overloads the sub-plugin calls exist.
 * - TensorShape::operator[] is const and hands out a mutable reference, as the
 *   one of the SDK does; the sub-plugin relies on that to overwrite a
 *   resizable dim.
 */
#ifndef __NNS_SNPE_MOCK_API_V1_HPP__
#define __NNS_SNPE_MOCK_API_V1_HPP__

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace zdl
{
namespace DlSystem
{

/** @brief Runtime processors an SNPE instance can be built for. */
enum class Runtime_t {
  CPU = 0,
  GPU = 1,
  DSP = 2,
  AIP_FIXED8_TF = 3,
  UNSET = 8
};

/** @brief Version of the SDK. */
struct Version_t {
  int32_t Major; /**< major number */
  int32_t Minor; /**< minor number */
  int32_t Teeny; /**< teeny number */

  /** @brief Get the version as a string. */
  std::string asString () const;
};

/**
 * @brief Optional value, as returned by the getters of the SDK.
 */
template <typename T> class Optional
{
  public:
  /** @brief Construct an unset value. */
  Optional () : m_has (false), m_value ()
  {
  }
  /** @brief Construct a set value. */
  Optional (const T &value) : m_has (true), m_value (value)
  {
  }

  /** @brief Tell whether the value is set. */
  explicit operator bool () const
  {
    return m_has;
  }
  /** @brief Get the value. */
  const T &operator* () const
  {
    return m_value;
  }
  /** @brief Get the value for modification. */
  T &operator* ()
  {
    return m_value;
  }
  /** @brief Convert to the held value. */
  operator const T & () const
  {
    return m_value;
  }

  private:
  bool m_has;
  T m_value;
};

/**
 * @brief Ordered list of strings, iterable over const char *.
 */
class StringList
{
  public:
  /** @brief Construct an empty list. */
  StringList ();
  /** @brief Copy a list. */
  StringList (const StringList &other);
  /** @brief Replace this list with a copy of another. */
  StringList &operator= (const StringList &other);

  /** @brief Append a string to the list. */
  void append (const char *str);
  /** @brief Get the number of strings in the list. */
  size_t size () const;
  /** @brief Get the string at the given index. */
  const char *at (size_t idx) const;
  /** @brief Get the first string of the list. */
  const char *const *begin () const;
  /** @brief Get the position past the last string of the list. */
  const char *const *end () const;

  private:
  /** @brief Point the iterable view at the stored strings. */
  void rebuild ();

  std::vector<std::string> m_items;
  std::vector<const char *> m_ptrs;
};

/**
 * @brief Shape of a tensor.
 */
class TensorShape
{
  public:
  /** @brief Construct an empty shape. */
  TensorShape ();
  /** @brief Construct a shape from the given dimensions. */
  TensorShape (const std::vector<size_t> &dims);

  /** @brief Get the rank of the shape. */
  size_t rank () const;
  /** @brief Get the dimension at the given index, for reading or writing. */
  size_t &operator[] (size_t idx) const;

  private:
  mutable std::vector<size_t> m_dims;
};

/**
 * @brief Attributes of the buffer of one tensor.
 */
class IBufferAttributes
{
  public:
  /** @brief Construct attributes of the given shape. */
  explicit IBufferAttributes (const TensorShape &dims);
  /** @brief Release the attributes. */
  virtual ~IBufferAttributes ();

  /** @brief Get the shape of the buffer. */
  virtual TensorShape getDims () const;

  private:
  TensorShape m_dims;
};

/**
 * @brief Tensor owned by the SDK, holding float elements.
 */
class ITensor
{
  public:
  /** @brief Construct a tensor of the given element count. */
  explicit ITensor (size_t elements);
  /** @brief Release the tensor. */
  virtual ~ITensor ();

  /** @brief Get the number of elements of the tensor. */
  size_t getSize () const;
  /** @brief Get the first element of the tensor. */
  float *begin ();
  /** @brief Get the first element of the tensor, for reading. */
  const float *cbegin () const;
  /** @brief Get the position past the last element of the tensor. */
  const float *cend () const;

  private:
  std::vector<float> m_data;
};

/**
 * @brief Factory creating tensors.
 */
class ITensorFactory
{
  public:
  /** @brief Release the factory. */
  virtual ~ITensorFactory ();

  /** @brief Create a tensor of the given shape. */
  virtual std::unique_ptr<ITensor> createTensor (const TensorShape &shape);
};

/**
 * @brief Map from a tensor name to a tensor the caller owns.
 */
class TensorMap
{
  public:
  /** @brief Put a tensor into the map. */
  void add (const char *name, ITensor *tensor);
  /** @brief Drop every entry of the map. */
  void clear ();
  /** @brief Get the tensor of the given name. */
  ITensor *getTensor (const char *name) const;

  private:
  std::map<std::string, ITensor *> m_tensors;
};

/**
 * @brief Encoding of the elements of a user supplied buffer.
 */
class UserBufferEncoding
{
  public:
  /** @brief Release the encoding. */
  virtual ~UserBufferEncoding ();
};

/**
 * @brief Encoding of a user supplied buffer holding float elements.
 */
class UserBufferEncodingFloat : public UserBufferEncoding
{
  public:
  /** @brief Release the encoding. */
  virtual ~UserBufferEncodingFloat ();
};

/**
 * @brief Buffer supplied by the caller, holding float elements.
 */
class IUserBuffer
{
  public:
  /** @brief Construct a buffer description of the given byte size. */
  explicit IUserBuffer (size_t size);
  /** @brief Release the buffer description. */
  virtual ~IUserBuffer ();

  /** @brief Point the buffer description at the given memory. */
  virtual bool setBufferAddress (void *buffer);
  /** @brief Get the byte size of the buffer. */
  size_t getSize () const;
  /** @brief Get the memory the buffer description points at. */
  void *getBufferAddress () const;

  private:
  void *m_address;
  size_t m_size;
};

/**
 * @brief Factory creating user supplied buffer descriptions.
 */
class IUserBufferFactory
{
  public:
  /** @brief Release the factory. */
  virtual ~IUserBufferFactory ();

  /** @brief Create a description of a user supplied buffer. */
  virtual std::unique_ptr<IUserBuffer> createUserBuffer (void *buffer, size_t bufSize,
      const TensorShape &strides, UserBufferEncoding *userBufferEncoding);
};

/**
 * @brief Map from a tensor name to a user supplied buffer.
 */
class UserBufferMap
{
  public:
  /** @brief Put a buffer description into the map. */
  void add (const char *name, IUserBuffer *buffer);
  /** @brief Drop every entry of the map. */
  void clear ();
  /** @brief Get the buffer description of the given name. */
  IUserBuffer *getUserBuffer (const char *name) const;

  private:
  std::map<std::string, IUserBuffer *> m_buffers;
};

/**
 * @brief Ordered list of runtimes an SNPE instance may run on.
 */
class RuntimeList
{
  public:
  /** @brief Construct an empty list. */
  RuntimeList ();
  /** @brief Construct a list holding the given runtime. */
  RuntimeList (Runtime_t runtime);

  /** @brief Drop every runtime of the list. */
  void clear ();
  /** @brief Add a runtime to the list. */
  bool add (Runtime_t runtime);
  /** @brief Get the number of runtimes of the list. */
  size_t size () const;
  /** @brief Get the runtime at the given index. */
  Runtime_t operator[] (size_t idx) const;

  private:
  std::vector<Runtime_t> m_runtimes;
};

} /* namespace DlSystem */

namespace DlContainer
{

/**
 * @brief An opened .dlc container.
 */
class IDlContainer
{
  public:
  /** @brief Release the container. */
  virtual ~IDlContainer ();

  /** @brief Open a mock container for the given model file. */
  static std::unique_ptr<IDlContainer> open (const std::string &filename) noexcept;

  /** @brief Tell whether the model has a resizable dim. */
  bool isResizable () const
  {
    return m_resizable;
  }
  /** @brief Tell whether the input tensor outgrows its buffer attributes. */
  bool hasOversizedInput () const
  {
    return m_oversized_input;
  }
  /** @brief Get the number of elements the emulated runtime writes. */
  size_t outputElements () const
  {
    return m_output_elements;
  }

  private:
  bool m_resizable;
  bool m_oversized_input;
  size_t m_output_elements;
};

} /* namespace DlContainer */

namespace SNPE
{

/**
 * @brief A built network, ready to execute.
 */
class SNPE
{
  public:
  /** @brief Construct a network over the given model properties. */
  SNPE (bool resizable, bool oversized_input, size_t output_elements);
  /** @brief Release the network. */
  virtual ~SNPE ();

  /** @brief Get the input tensor names of the network. */
  DlSystem::Optional<DlSystem::StringList> getInputTensorNames () const;
  /** @brief Get the output tensor names of the network. */
  DlSystem::Optional<DlSystem::StringList> getOutputTensorNames () const;
  /** @brief Get the shape of the named input tensor. */
  DlSystem::Optional<DlSystem::TensorShape> getInputDimensions (const char *name) const;
  /** @brief Get the buffer attributes of the named tensor, owned by the network. */
  DlSystem::Optional<DlSystem::IBufferAttributes *>
  getInputOutputBufferAttributes (const char *name) const;

  /** @brief Run the emulated model over the given tensor maps. */
  bool execute (const DlSystem::TensorMap &input, DlSystem::TensorMap &output);
  /** @brief Run the emulated model over the given tensor maps. */
  bool execute (const DlSystem::UserBufferMap &input, const DlSystem::UserBufferMap &output);

  private:
  bool m_resizable;
  bool m_oversized_input;
  size_t m_output_elements;
  mutable std::map<std::string, std::unique_ptr<DlSystem::IBufferAttributes>> m_attributes;
  std::map<std::string, std::unique_ptr<DlSystem::ITensor>> m_outputs;
};

/**
 * @brief Builder of an SNPE network.
 */
class SNPEBuilder
{
  public:
  /** @brief Construct a builder over the given container. */
  explicit SNPEBuilder (DlContainer::IDlContainer *container);

  /** @brief Accept the output tensor names. */
  SNPEBuilder &setOutputTensors (const DlSystem::StringList &outputTensors);
  /** @brief Accept the user supplied buffer mode. */
  SNPEBuilder &setUseUserSuppliedBuffers (bool useUserSuppliedBuffers);
  /** @brief Accept the init cache mode. */
  SNPEBuilder &setInitCacheMode (bool cacheMode);
  /** @brief Accept the runtime order. */
  SNPEBuilder &setRuntimeProcessorOrder (const DlSystem::RuntimeList &runtimeList);
  /** @brief Accept the CPU fallback mode. */
  SNPEBuilder &setCPUFallbackMode (bool cpuFallback);
  /** @brief Build the network. */
  std::unique_ptr<SNPE> build () noexcept;

  private:
  DlContainer::IDlContainer *m_container;
};

/**
 * @brief Entry points of the SDK that do not belong to a network.
 */
class SNPEFactory
{
  public:
  /** @brief Tell whether the given runtime is available; only the CPU one is. */
  static bool isRuntimeAvailable (DlSystem::Runtime_t runtime);
  /** @brief Get the factory creating tensors. */
  static DlSystem::ITensorFactory &getTensorFactory ();
  /** @brief Get the factory creating user supplied buffer descriptions. */
  static DlSystem::IUserBufferFactory &getUserBufferFactory ();
  /** @brief Get the version of the mock SDK. */
  static DlSystem::Version_t getLibraryVersion ();
};

} /* namespace SNPE */
} /* namespace zdl */

#endif /* __NNS_SNPE_MOCK_API_V1_HPP__ */
