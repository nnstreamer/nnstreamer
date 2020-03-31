/**
 * @file        dummy_edgetpu.h
 * @date        17 Dec 2019
 * @brief       Dummy implementation of tflite and edgetpu for unit tests.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @brief	This header provides a dummy edgetpu library that redirects
 *		requests to CPU backends.
 */
#include <iostream>
#include <cassert>
#include "edgetpu.hh"
#include <stdio.h>

namespace edgetpu {

static EdgeTpuManagerDummy manager;

/** @brief Dummy edgetpu get function */
EdgeTpuContext *EdgeTpuContextDummy::get()
{
  return this;
}

/** @brief Dummy singleton methos */
EdgeTpuManager * EdgeTpuManager::GetSingleton()
{
  return &manager;
}

/** @brief Dummy device open */
std::shared_ptr<EdgeTpuContext> EdgeTpuManagerDummy::OpenDevice()
{
  std::shared_ptr<EdgeTpuContext> ret(new EdgeTpuContextDummy());
  return ret;
}

static TfLiteRegistration nullreg = { 0 };

/** @brief Dummy tflite custom op */
TfLiteRegistration * RegisterCustomOp()
{
  return &nullreg;
}

/** @brief virutal destructor */
EdgeTpuContext::~EdgeTpuContext()
{
}

}; /* namespace edgetpu */
