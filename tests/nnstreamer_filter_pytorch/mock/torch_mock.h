/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        torch_mock.h
 * @date        27 Oct 2026
 * @brief       Control and observation of the LibTorch mock
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * A mock model file holds one keyword that tells what torch::jit::load() and
 * Module::forward() do with it:
 *  - identity: return the first input
 *  - tuple, list: return all inputs as a tuple or a list
 *  - nested: return a tuple of the first input and a list of the others
 *  - none: return a value that is not a tensor, tuple or list
 *  - bool: return a bool tensor of the shape of the first input
 *  - grow: return a tensor of the type of the first input, twice as large
 *  - empty: return a tensor whose first dimension is 0
 *  - scalar: return a 0-dim tensor of the type of the first input
 *  - throw_runtime, throw_exception, throw_unknown: forward() throws a
 *    std::runtime_error, a std::exception or an int
 *  - load_invalid_argument, load_unknown: load() throws a
 *    std::invalid_argument or an int
 * Any other content, or a missing file, makes load() throw a runtime_error.
 */
#ifndef __NNS_TORCH_MOCK_H__
#define __NNS_TORCH_MOCK_H__

#include <torch/script.h>

namespace torch_mock
{
/** @brief Reset the counters and the knobs to their defaults. */
void reset ();
/** @brief Whether load() accepts a CUDA device (default: false). */
void setCudaAvailable (bool available);
/** @brief The number of Module::eval() calls. */
int evalCount ();
/** @brief The number of Module::forward() calls. */
int forwardCount ();
/** @brief The number of successful load() calls. */
int loadCount ();
/** @brief The device type of the last load() call. */
c10::DeviceType lastLoadDevice ();
/** @brief The tensors the last forward() call received. */
const std::vector<at::Tensor> &lastInputs ();
} /* namespace torch_mock */

#endif /* __NNS_TORCH_MOCK_H__ */
