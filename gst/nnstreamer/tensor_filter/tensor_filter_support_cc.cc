/**
 * GStreamer Tensor_filter, C++ Subplugin Support. (this is not a subplugin)
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_filter_support_cc.cc
 * @date	22 Jan 2020
 * @brief	Base class for tensor_filter subplugins of C++ classes.
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @details
 *    This is not a subplugin, but a helper for C++ subplugins.
 *    If you want to write a wrapper for neural network frameworks
 *    or a hardware adaptor in C++, this is what you want.
 *    If you want to attach a single C++ class/object as a filter
 *    (a.k.a. custom filter) to a pipeline, use tensor_filter_cpp
 */

#include <string.h>
#include <assert.h>
#include <errno.h>

#if __cplusplus < 201103L
#warn C++11 is required for safe execution
#else
#include <type_traits>
#endif

#define __NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef __NO_ANONYMOUS_NESTED_STRUCT

#include "tensor_filter_support_cc.hh"

namespace nnstreamer {

/******************************************************
 ** Class methods of tensor_filter_subplugin (base)  **
 ******************************************************/

#define _SANITY_CHECK (0xFACE217714DEADE7ULL)

/**
 * @brief C tensor-filter wrapper callback function, "open"
 */
int tensor_filter_subplugin::cpp_open (const GstTensorFilterProperties * prop,
    void **private_data)
{
  const GstTensorFilterFramework * tfsp = nnstreamer_filter_find (prop->fwname);

  assert (tfsp);
  assert (tfsp->version == GST_TENSOR_FILTER_FRAMEWORK_V1);

  /* 1. Fetch stored empty object from subplugin api (subplugin_data) */
  tensor_filter_subplugin *sp = (tensor_filter_subplugin *) tfsp->v1.subplugin_data;
  assert (sp->sanity == _SANITY_CHECK); /** tfsp is using me! */

  /* 2. Spawn another empty object and configure the empty object */
  tensor_filter_subplugin &obj = sp->getEmptyInstance();
  obj.configure_instance (prop);

  /* 3. Mark that this is not a representative (found by nnstreamer_filter_find) empty object */
  obj.fwdesc.v1.subplugin_data = nullptr;

  /* 4. Save the object as *private_data */
  *private_data = &obj;

  return 0;
}

#define start_up(t, p) \
  tensor_filter_subplugin *t = (tensor_filter_subplugin *) p; \
  assert (t); \
  assert (t->sanity == _SANITY_CHECK); \
  assert (t->fwdesc.v1.subplugin_data == nullptr);

/**
 * @brief C tensor-filter wrapper callback function, "close"
 */
void tensor_filter_subplugin::cpp_close (const GstTensorFilterProperties * prop,
    void **private_data)
{
  start_up (obj, *private_data);

  *private_data = nullptr;
  delete obj;
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "invoke"
 */
int tensor_filter_subplugin::cpp_invoke (const GstTensorFilterProperties *prop,
    void *private_data, const GstTensorMemory *input,
    GstTensorMemory *output)
{
  start_up (obj, private_data);
  return obj->invoke (*input, *output);
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "getFrameworkInfo"
 */
int tensor_filter_subplugin::cpp_getFrameworkInfo (const GstTensorFilterProperties * prop,
    void *private_data, GstTensorFilterFrameworkInfo *fw_info)
{
  start_up (obj, private_data);
  return obj->getFrameworkInfo (*fw_info);
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "getModelInfo"
 */
int tensor_filter_subplugin::cpp_getModelInfo (const GstTensorFilterProperties * prop,
    void *private_data, model_info_ops ops,
    GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  start_up (obj, private_data);
  return obj->getModelInfo (ops, *in_info, *out_info);
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "eventHandler"
 */
int tensor_filter_subplugin::cpp_eventHandler (const GstTensorFilterProperties * prop,
    void *private_data, event_ops ops, GstTensorFilterFrameworkEventData *data)
{
  start_up (obj, private_data);
  return obj->eventHandler (ops, *data);
}

/**
 * @brief The template for fwdesc, the C wrapper (V1) struct.
 */
const GstTensorFilterFramework tensor_filter_subplugin::fwdesc_template = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V1,
  .open = cpp_open,
  .close = cpp_close,
  {
    .v1 = {
      .invoke = cpp_invoke,
      .getFrameworkInfo = cpp_getFrameworkInfo,
      .getModelInfo = cpp_getModelInfo,
      .eventHandler = cpp_eventHandler,
      .subplugin_data = nullptr,
    }
  }
};

/**
 * @brief Register the subplugin "derived" class.
 * @detail A derived class MUST register itself with this function in order
 *         to be available for nnstreamer pipelines, i.e., at its init().
 *         The derived class type should be the template typename.
 * @retval Returns an "emptyInstnace" of the derived class. It is recommended
 *         to keep the object and feed to the unregister function.
 */
template<typename T>
T * tensor_filter_subplugin::register_subplugin ()
{
#if __cplusplus < 201103L
#warn C++11 is required for safe execution
#else
  /** The given class T should be derived from tensor_filter_subplugin */
  assert ((std::is_base_of<tensor_filter_subplugin, T>::value));
#endif

  T *emptyInstance = T::T();

  assert (emptyInstance);

  memcpy (&emptyInstance->fwdesc, &fwdesc_template, sizeof (fwdesc_template));
  emptyInstance->fwdesc.subplugin_data = emptyInstance;

  nnstreamer_filter_probe (&emptyInstance->fwdesc);

  return emptyInstance;
}

/**
 * @brief Unregister the registered "derived" class.
 * @detail The registered derived class may unregister itself if it can
 *         guarantee that the class won't be used anymore; i.e., at its exit().
 *         The derived class type should be the template typename.
 * @param [in] emptyInstance An emptyInstance that mey be "delete"d by this
 *             function. It may be created by getEmptyInstance() or the one
 *             created by register_subplugin(); It is recommended to keep
 *             the object created by register_subplugin() and feed it to
 *             unregister_subplugin().
 */
template<typename T>
void tensor_filter_subplugin::unregister_subplugin (T * emptyInstance)
{
  GstTensorFilterFrameworkInfo info;
#if __cplusplus < 201103L
#warn C++11 is required for safe execution
#else
  /** The given class T should be derived from tensor_filter_subplugin */
  assert((std::is_base_of<tensor_filter_subplugin, T>::value));
#endif
  assert (emptyInstance);

  emptyInstance->getFrameworkInfo (&info);
  nnstreamer_filter_exit (info.name);

  delete emptyInstance;
}

/**
 * @brief Base constructor. The object represents a non-functional empty anchor
 */
tensor_filter_subplugin::tensor_filter_subplugin ():
    sanity(_SANITY_CHECK)
{
  memcpy (&fwdesc, &fwdesc_template, sizeof (fwdesc_template));
}

/**
 * @brief Base destructor. The object represents a single GST element instance in a pipeline
 */
tensor_filter_subplugin::~tensor_filter_subplugin ()
{
  /* Nothing to do */
}

/**
 * @brief Base eventHandler, which does nothing!
 */
int tensor_filter_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return 0;
}

} /* namespace nnstreamer */
