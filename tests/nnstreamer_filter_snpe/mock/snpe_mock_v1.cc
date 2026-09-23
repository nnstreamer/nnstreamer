/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    snpe_mock_v1.cc
 * @date    22 Sep 2026
 * @brief   Mock implementation of the SNPE 1.x C++ API.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * The emulated model has one input and one output named by SNPE_MOCK_INPUT_NAME
 * and SNPE_MOCK_OUTPUT_NAME, and computes output = input + SNPE_MOCK_ADDEND in
 * float. A container file is never read: its name decides the shape of the
 * model, see snpe_mock_model_load().
 */

#include <algorithm>
#include <string>
#include <utility>

#include <glib.h>

#include "snpe_mock.h"

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/IUserBufferFactory.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <DlSystem/TensorMap.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>

namespace zdl
{
namespace DlSystem
{

/**
 * @brief Get the version as a string.
 */
std::string
Version_t::asString () const
{
  return std::to_string (Major) + "." + std::to_string (Minor) + "."
         + std::to_string (Teeny);
}

/**
 * @brief Construct an empty list.
 */
StringList::StringList ()
{
}

/**
 * @brief Copy a list.
 */
StringList::StringList (const StringList &other) : m_items (other.m_items)
{
  rebuild ();
}

/**
 * @brief Replace this list with a copy of another.
 */
StringList &
StringList::operator= (const StringList &other)
{
  if (this != &other) {
    m_items = other.m_items;
    rebuild ();
  }

  return *this;
}

/**
 * @brief Point the iterable view at the stored strings.
 */
void
StringList::rebuild ()
{
  m_ptrs.clear ();
  for (const std::string &item : m_items)
    m_ptrs.push_back (item.c_str ());
}

/**
 * @brief Append a string to the list.
 */
void
StringList::append (const char *str)
{
  if (!str)
    return;

  m_items.push_back (str);
  rebuild ();
}

/**
 * @brief Get the number of strings in the list.
 */
size_t
StringList::size () const
{
  return m_items.size ();
}

/**
 * @brief Get the string at the given index.
 */
const char *
StringList::at (size_t idx) const
{
  return (idx < m_items.size ()) ? m_items[idx].c_str () : nullptr;
}

/**
 * @brief Get the first string of the list.
 */
const char *const *
StringList::begin () const
{
  return m_ptrs.data ();
}

/**
 * @brief Get the position past the last string of the list.
 */
const char *const *
StringList::end () const
{
  return m_ptrs.data () + m_ptrs.size ();
}

/**
 * @brief Construct an empty shape.
 */
TensorShape::TensorShape ()
{
}

/**
 * @brief Construct a shape from the given dimensions.
 */
TensorShape::TensorShape (const std::vector<size_t> &dims) : m_dims (dims)
{
}

/**
 * @brief Get the rank of the shape.
 */
size_t
TensorShape::rank () const
{
  return m_dims.size ();
}

/**
 * @brief Get the dimension at the given index, for reading or writing.
 */
size_t &
TensorShape::operator[] (size_t idx) const
{
  g_assert (idx < m_dims.size ());
  return m_dims[idx];
}

/**
 * @brief Construct attributes of the given shape.
 */
IBufferAttributes::IBufferAttributes (const TensorShape &dims) : m_dims (dims)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES);
}

/**
 * @brief Release the attributes.
 */
IBufferAttributes::~IBufferAttributes ()
{
  snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES);
}

/**
 * @brief Get the shape of the buffer.
 */
TensorShape
IBufferAttributes::getDims () const
{
  return m_dims;
}

/**
 * @brief Construct a tensor of the given element count.
 */
ITensor::ITensor (size_t elements) : m_data (elements, 0.0f)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_TENSOR);
}

/**
 * @brief Release the tensor.
 */
ITensor::~ITensor ()
{
  snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_TENSOR);
}

/**
 * @brief Get the number of elements of the tensor.
 */
size_t
ITensor::getSize () const
{
  return m_data.size ();
}

/**
 * @brief Get the first element of the tensor.
 */
float *
ITensor::begin ()
{
  return m_data.data ();
}

/**
 * @brief Get the first element of the tensor, for reading.
 */
const float *
ITensor::cbegin () const
{
  return m_data.data ();
}

/**
 * @brief Get the position past the last element of the tensor.
 */
const float *
ITensor::cend () const
{
  return m_data.data () + m_data.size ();
}

/**
 * @brief Release the factory.
 */
ITensorFactory::~ITensorFactory ()
{
}

/**
 * @brief Create a tensor of the given shape.
 */
std::unique_ptr<ITensor>
ITensorFactory::createTensor (const TensorShape &shape)
{
  size_t elements = (shape.rank () == 0) ? 0 : 1;

  for (size_t i = 0; i < shape.rank (); i++)
    elements *= shape[i];

  return std::unique_ptr<ITensor> (new ITensor (elements));
}

/**
 * @brief Put a tensor into the map.
 */
void
TensorMap::add (const char *name, ITensor *tensor)
{
  if (name)
    m_tensors[name] = tensor;
}

/**
 * @brief Drop every entry of the map.
 */
void
TensorMap::clear ()
{
  m_tensors.clear ();
}

/**
 * @brief Get the tensor of the given name.
 */
ITensor *
TensorMap::getTensor (const char *name) const
{
  if (!name)
    return nullptr;

  auto it = m_tensors.find (name);
  return (it == m_tensors.end ()) ? nullptr : it->second;
}

/**
 * @brief Release the encoding.
 */
UserBufferEncoding::~UserBufferEncoding ()
{
}

/**
 * @brief Release the encoding.
 */
UserBufferEncodingFloat::~UserBufferEncodingFloat ()
{
}

/**
 * @brief Construct a buffer description of the given byte size.
 */
IUserBuffer::IUserBuffer (size_t size) : m_address (nullptr), m_size (size)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_USER_BUFFER);
}

/**
 * @brief Release the buffer description.
 */
IUserBuffer::~IUserBuffer ()
{
  snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_USER_BUFFER);
}

/**
 * @brief Point the buffer description at the given memory.
 */
bool
IUserBuffer::setBufferAddress (void *buffer)
{
  m_address = buffer;
  return true;
}

/**
 * @brief Get the byte size of the buffer.
 */
size_t
IUserBuffer::getSize () const
{
  return m_size;
}

/**
 * @brief Get the memory the buffer description points at.
 */
void *
IUserBuffer::getBufferAddress () const
{
  return m_address;
}

/**
 * @brief Release the factory.
 */
IUserBufferFactory::~IUserBufferFactory ()
{
}

/**
 * @brief Create a description of a user supplied buffer.
 */
std::unique_ptr<IUserBuffer>
IUserBufferFactory::createUserBuffer (void *buffer, size_t bufSize,
    const TensorShape &strides, UserBufferEncoding *userBufferEncoding)
{
  std::unique_ptr<IUserBuffer> user_buffer (new IUserBuffer (bufSize));

  (void) strides;
  (void) userBufferEncoding;
  user_buffer->setBufferAddress (buffer);

  return user_buffer;
}

/**
 * @brief Put a buffer description into the map.
 */
void
UserBufferMap::add (const char *name, IUserBuffer *buffer)
{
  if (name)
    m_buffers[name] = buffer;
}

/**
 * @brief Drop every entry of the map.
 */
void
UserBufferMap::clear ()
{
  m_buffers.clear ();
}

/**
 * @brief Get the buffer description of the given name.
 */
IUserBuffer *
UserBufferMap::getUserBuffer (const char *name) const
{
  if (!name)
    return nullptr;

  auto it = m_buffers.find (name);
  return (it == m_buffers.end ()) ? nullptr : it->second;
}

/**
 * @brief Construct an empty list.
 */
RuntimeList::RuntimeList ()
{
}

/**
 * @brief Construct a list holding the given runtime.
 */
RuntimeList::RuntimeList (Runtime_t runtime)
{
  m_runtimes.push_back (runtime);
}

/**
 * @brief Drop every runtime of the list.
 */
void
RuntimeList::clear ()
{
  m_runtimes.clear ();
}

/**
 * @brief Add a runtime to the list.
 */
bool
RuntimeList::add (Runtime_t runtime)
{
  m_runtimes.push_back (runtime);
  return true;
}

/**
 * @brief Get the number of runtimes of the list.
 */
size_t
RuntimeList::size () const
{
  return m_runtimes.size ();
}

/**
 * @brief Get the runtime at the given index.
 */
Runtime_t
RuntimeList::operator[] (size_t idx) const
{
  return (idx < m_runtimes.size ()) ? m_runtimes[idx] : Runtime_t::UNSET;
}

} /* namespace DlSystem */

namespace DlContainer
{

/**
 * @brief Release the container.
 */
IDlContainer::~IDlContainer ()
{
  snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_CONTAINER);
}

/**
 * @brief Open a mock container for the given model file.
 */
std::unique_ptr<IDlContainer>
IDlContainer::open (const std::string &filename) noexcept
{
  snpe_mock_model model;

  if (!snpe_mock_model_load (filename.c_str (), &model))
    return nullptr;

  std::unique_ptr<IDlContainer> container (new IDlContainer ());
  container->m_resizable = model.resizable;
  container->m_oversized_input = model.oversized_input;
  container->m_output_elements = model.output_elements;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_CONTAINER);

  return container;
}

} /* namespace DlContainer */

namespace SNPE
{

namespace
{

/**
 * @brief Get the shape of every tensor of the emulated model.
 *
 * A resizable model gets a rank of two with the resizable dim last, because
 * the sub-plugin only inspects the dims above index zero while it lays out a
 * user buffer.
 */
std::vector<size_t>
model_dims (bool resizable)
{
  std::vector<size_t> dims (1, 1U);

  if (resizable)
    dims.push_back (0U);

  return dims;
}

} /* namespace */

/**
 * @brief Construct a network over the given model properties.
 */
SNPE::SNPE (bool resizable, bool oversized_input, size_t output_elements)
    : m_resizable (resizable), m_oversized_input (oversized_input),
      m_output_elements (output_elements)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_SNPE);
}

/**
 * @brief Release the network.
 */
SNPE::~SNPE ()
{
  snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_SNPE);
}

/**
 * @brief Get the input tensor names of the network.
 */
DlSystem::Optional<DlSystem::StringList>
SNPE::getInputTensorNames () const
{
  DlSystem::StringList names;

  names.append (SNPE_MOCK_INPUT_NAME);
  return DlSystem::Optional<DlSystem::StringList> (names);
}

/**
 * @brief Get the output tensor names of the network.
 */
DlSystem::Optional<DlSystem::StringList>
SNPE::getOutputTensorNames () const
{
  DlSystem::StringList names;

  names.append (SNPE_MOCK_OUTPUT_NAME);
  return DlSystem::Optional<DlSystem::StringList> (names);
}

/**
 * @brief Get the shape of the named input tensor.
 */
DlSystem::Optional<DlSystem::TensorShape>
SNPE::getInputDimensions (const char *name) const
{
  if (g_strcmp0 (name, SNPE_MOCK_INPUT_NAME) != 0)
    return DlSystem::Optional<DlSystem::TensorShape> ();

  /* An oversized input reports a shape its buffer attributes do not. */
  std::vector<size_t> dims = model_dims (m_resizable);
  if (m_oversized_input)
    dims[dims.size () - 1] = SNPE_MOCK_OVERSIZED_INPUT_ELEMENTS;

  return DlSystem::Optional<DlSystem::TensorShape> (DlSystem::TensorShape (dims));
}

/**
 * @brief Get the buffer attributes of the named tensor, owned by the network.
 */
DlSystem::Optional<DlSystem::IBufferAttributes *>
SNPE::getInputOutputBufferAttributes (const char *name) const
{
  if (!name)
    return DlSystem::Optional<DlSystem::IBufferAttributes *> ();

  if (g_strcmp0 (name, SNPE_MOCK_INPUT_NAME) != 0
      && g_strcmp0 (name, SNPE_MOCK_OUTPUT_NAME) != 0)
    return DlSystem::Optional<DlSystem::IBufferAttributes *> ();

  auto it = m_attributes.find (name);
  if (it == m_attributes.end ()) {
    DlSystem::TensorShape shape (model_dims (m_resizable));
    it = m_attributes
             .insert (std::make_pair (std::string (name),
                 std::unique_ptr<DlSystem::IBufferAttributes> (
                     new DlSystem::IBufferAttributes (shape))))
             .first;
  }

  return DlSystem::Optional<DlSystem::IBufferAttributes *> (it->second.get ());
}

/**
 * @brief Run the emulated model over the given tensor maps.
 */
bool
SNPE::execute (const DlSystem::TensorMap &input, DlSystem::TensorMap &output)
{
  const DlSystem::ITensor *in = input.getTensor (SNPE_MOCK_INPUT_NAME);

  std::unique_ptr<DlSystem::ITensor> out (new DlSystem::ITensor (m_output_elements));
  float *dest = out->begin ();

  for (size_t i = 0; i < m_output_elements; i++) {
    const float value = (in && i < in->getSize ()) ? in->cbegin ()[i] : 0.0f;
    dest[i] = value + SNPE_MOCK_ADDEND;
  }

  if (snpe_mock_get_failure () != SNPE_MOCK_FAIL_EXECUTE_NO_OUTPUT) {
    output.add (SNPE_MOCK_OUTPUT_NAME, out.get ());
    m_outputs[SNPE_MOCK_OUTPUT_NAME] = std::move (out);
  }

  return true;
}

/**
 * @brief Run the emulated model over the given user buffer maps.
 */
bool
SNPE::execute (const DlSystem::UserBufferMap &input, const DlSystem::UserBufferMap &output)
{
  const DlSystem::IUserBuffer *in = input.getUserBuffer (SNPE_MOCK_INPUT_NAME);
  DlSystem::IUserBuffer *out = output.getUserBuffer (SNPE_MOCK_OUTPUT_NAME);

  if (!out || !out->getBufferAddress ())
    return false;

  const size_t out_elems = out->getSize () / sizeof (float);
  const size_t in_elems
      = (in && in->getBufferAddress ()) ? in->getSize () / sizeof (float) : 0;
  const float *src = in ? static_cast<const float *> (in->getBufferAddress ()) : nullptr;
  float *dest = static_cast<float *> (out->getBufferAddress ());

  for (size_t i = 0; i < out_elems; i++)
    dest[i] = ((src && i < in_elems) ? src[i] : 0.0f) + SNPE_MOCK_ADDEND;

  return true;
}

/**
 * @brief Construct a builder over the given container.
 */
SNPEBuilder::SNPEBuilder (DlContainer::IDlContainer *container)
    : m_container (container)
{
}

/**
 * @brief Accept the output tensor names.
 */
SNPEBuilder &
SNPEBuilder::setOutputTensors (const DlSystem::StringList &outputTensors)
{
  (void) outputTensors;
  return *this;
}

/**
 * @brief Accept the user supplied buffer mode.
 */
SNPEBuilder &
SNPEBuilder::setUseUserSuppliedBuffers (bool useUserSuppliedBuffers)
{
  (void) useUserSuppliedBuffers;
  return *this;
}

/**
 * @brief Accept the init cache mode.
 */
SNPEBuilder &
SNPEBuilder::setInitCacheMode (bool cacheMode)
{
  (void) cacheMode;
  return *this;
}

/**
 * @brief Accept the runtime order.
 */
SNPEBuilder &
SNPEBuilder::setRuntimeProcessorOrder (const DlSystem::RuntimeList &runtimeList)
{
  (void) runtimeList;
  return *this;
}

/**
 * @brief Accept the CPU fallback mode.
 */
SNPEBuilder &
SNPEBuilder::setCPUFallbackMode (bool cpuFallback)
{
  (void) cpuFallback;
  return *this;
}

/**
 * @brief Build the network.
 */
std::unique_ptr<SNPE>
SNPEBuilder::build () noexcept
{
  if (!m_container || snpe_mock_get_failure () == SNPE_MOCK_FAIL_BUILD)
    return nullptr;

  return std::unique_ptr<SNPE> (new SNPE (m_container->isResizable (),
      m_container->hasOversizedInput (), m_container->outputElements ()));
}

/**
 * @brief Tell whether the given runtime is available; only the CPU one is.
 */
bool
SNPEFactory::isRuntimeAvailable (DlSystem::Runtime_t runtime)
{
  return runtime == DlSystem::Runtime_t::CPU;
}

/**
 * @brief Get the factory creating tensors.
 */
DlSystem::ITensorFactory &
SNPEFactory::getTensorFactory ()
{
  static DlSystem::ITensorFactory factory;

  return factory;
}

/**
 * @brief Get the factory creating user supplied buffer descriptions.
 */
DlSystem::IUserBufferFactory &
SNPEFactory::getUserBufferFactory ()
{
  static DlSystem::IUserBufferFactory factory;

  return factory;
}

/**
 * @brief Get the version of the mock SDK.
 */
DlSystem::Version_t
SNPEFactory::getLibraryVersion ()
{
  DlSystem::Version_t version = { 1, 68, 0 };

  return version;
}

} /* namespace SNPE */
} /* namespace zdl */
