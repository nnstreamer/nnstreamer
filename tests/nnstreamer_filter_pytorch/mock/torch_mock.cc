/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        torch_mock.cc
 * @date        27 Oct 2026
 * @brief       Implementation of the LibTorch mock
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <glib.h>

#include "torch_mock.h"

namespace
{
bool cuda_available = false;
bool cuda_fails = false;
int eval_count = 0;
int forward_count = 0;
int load_count = 0;
c10::DeviceType last_load_device = c10::DeviceType::CPU;
std::vector<at::Tensor> last_inputs;

/**
 * @brief Read the behaviour keyword of a mock model file.
 */
std::string
readBehaviour (const std::string &filename)
{
  gchar *content = NULL;

  if (!g_file_get_contents (filename.c_str (), &content, NULL, NULL))
    throw std::runtime_error ("open file failed: " + filename);

  std::string behaviour (g_strstrip (content));
  g_free (content);
  return behaviour;
}

/**
 * @brief Whether the given keyword names a behaviour of forward().
 */
bool
isForwardBehaviour (const std::string &behaviour)
{
  static const char *known[] = { "identity", "tuple", "list", "nested", "none",
    "bool", "grow", "empty", "scalar", "throw_runtime", "throw_exception",
    "throw_unknown", "throw_after_first" };

  for (const char *k : known) {
    if (behaviour == k)
      return true;
  }
  return false;
}
} /* namespace */

namespace torch_mock
{
/** @brief Reset the counters and the knobs to their defaults. */
void
reset ()
{
  cuda_available = false;
  cuda_fails = false;
  eval_count = forward_count = load_count = 0;
  last_load_device = c10::DeviceType::CPU;
  last_inputs.clear ();
}

/** @brief Whether load() accepts a CUDA device. */
void
setCudaAvailable (bool available)
{
  cuda_available = available;
}

/** @brief Whether Tensor::to() from or to CUDA throws. */
void
setCudaFails (bool fails)
{
  cuda_fails = fails;
}

/** @brief Whether moving a tensor to or from a CUDA device fails. */
bool
cudaFails ()
{
  return cuda_fails;
}

/** @brief The number of Module::eval() calls. */
int
evalCount ()
{
  return eval_count;
}

/** @brief The number of Module::forward() calls. */
int
forwardCount ()
{
  return forward_count;
}

/** @brief The number of successful load() calls. */
int
loadCount ()
{
  return load_count;
}

/** @brief The device type of the last load() call. */
c10::DeviceType
lastLoadDevice ()
{
  return last_load_device;
}

/** @brief The tensors the last forward() call received. */
const std::vector<at::Tensor> &
lastInputs ()
{
  return last_inputs;
}
} /* namespace torch_mock */

namespace torch
{
namespace jit
{
/** @brief Switch the module to evaluation mode. */
void
Module::eval ()
{
  eval_count++;
}

/** @brief Run the module as its behaviour says. */
IValue
Module::forward (std::vector<IValue> inputs)
{
  forward_count++;
  last_inputs.clear ();
  for (const IValue &input : inputs)
    last_inputs.push_back (input.toTensor ());

  if (behaviour_ == "throw_runtime")
    throw std::runtime_error ("mock runtime error");
  if (behaviour_ == "throw_exception")
    throw std::exception ();
  if (behaviour_ == "throw_unknown")
    throw 42;
  if (behaviour_ == "throw_after_first" && forward_count > 1)
    throw std::runtime_error ("mock runtime error after the first run");
  if (behaviour_ == "none" || inputs.empty ())
    return IValue ();

  const at::Tensor first = inputs[0].toTensor ();
  std::vector<int64_t> sizes (first.sizes ().begin (), first.sizes ().end ());

  if (behaviour_ == "tuple") {
    auto tuple = std::make_shared<c10::ivalue::Tuple> ();
    tuple->elements_ = inputs;
    return IValue (tuple);
  }
  if (behaviour_ == "list")
    return IValue::makeList (inputs);
  if (behaviour_ == "nested") {
    auto tuple = std::make_shared<c10::ivalue::Tuple> ();
    tuple->elements_.push_back (inputs[0]);
    tuple->elements_.push_back (
        IValue::makeList (std::vector<IValue> (inputs.begin () + 1, inputs.end ())));
    return IValue (tuple);
  }
  if (behaviour_ == "bool")
    return IValue (at::Tensor (sizes, c10::ScalarType::Bool));
  if (behaviour_ == "grow") {
    sizes[0] *= 2;
    return IValue (at::Tensor (sizes, first.scalar_type ()));
  }
  if (behaviour_ == "empty") {
    sizes[0] = 0;
    return IValue (at::Tensor (sizes, first.scalar_type ()));
  }
  if (behaviour_ == "scalar")
    return IValue (at::Tensor (std::vector<int64_t> (), first.scalar_type ()));

  return inputs[0];
}

/** @brief Load a mock module from the given file onto the given device. */
Module
load (const std::string &filename, c10::Device device)
{
  std::string behaviour = readBehaviour (filename);

  if (behaviour == "load_invalid_argument")
    throw std::invalid_argument ("mock invalid argument");
  if (behaviour == "load_unknown")
    throw 42;
  if (!isForwardBehaviour (behaviour))
    throw std::runtime_error ("not a mock model: " + filename);
  if (device.type () == c10::DeviceType::CUDA && !cuda_available)
    throw std::runtime_error ("no CUDA device");

  last_load_device = device.type ();
  load_count++;
  return Module (behaviour);
}
} /* namespace jit */
} /* namespace torch */
