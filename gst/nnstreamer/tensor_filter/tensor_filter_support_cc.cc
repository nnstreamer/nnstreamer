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
 * @see		http://github.com/nnstreamer/nnstreamer
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
#include <nnstreamer_log.h>

#include <system_error>

#define __NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef __NO_ANONYMOUS_NESTED_STRUCT

#include <nnstreamer_cppplugin_api_filter.hh>

namespace nnstreamer {

/******************************************************
 ** Class methods of tensor_filter_subplugin (base)  **
 ******************************************************/

#define _SANITY_CHECK (0xFACE217714DEADE7ULL)
#define _RETURN_ERR_WITH_MSG(c, m) \
  do { \
    nns_loge ("%s", m); \
    return c; \
  } while (0);

#define GET_TFSP_WITH_CHECKS(obj, private_data) \
  do { \
    try { \
      obj = get_tfsp_with_checks (private_data); \
    } catch (const std::exception &e) { \
      /** @todo Write exception handlers. */ \
      return -EINVAL; \
      /** @todo return different error codes according to exceptions */ \
    } \
  } while (0);

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
  try {
    obj.configure_instance (prop);
  } catch (const std::invalid_argument & e) {
    _RETURN_ERR_WITH_MSG (-EINVAL, e.what());
  } catch (const std::system_error & e) {
    _RETURN_ERR_WITH_MSG (e.code().value() * -1, e.what());
  } catch (const std::runtime_error & e) {
    /** @todo return different error codes according to exceptions */
    _RETURN_ERR_WITH_MSG (-1, e.what());
  } catch (const std::exception & e) {
    /** @todo Write exception handlers. */
    /** @todo return different error codes according to exceptions */
    _RETURN_ERR_WITH_MSG (-1, e.what());
  }

  /* 3. Mark that this is not a representative (found by nnstreamer_filter_find) empty object */
  obj.fwdesc.v1.subplugin_data = nullptr;

  /* 4. Save the object as *private_data */
#if __GNUC__ < 5 || __cpllusplus < 201103L
  *private_data = &(obj);
#else /* It is safer w/ addressof, but old gcc doesn't appear to support it */
  *private_data = std::addressof (obj);
#endif

  return 0;
}

/**
 * @brief Get tensor_filter_subplugin pointer with some sanity checks
 */
tensor_filter_subplugin * tensor_filter_subplugin::get_tfsp_with_checks (
    void * ptr)
{
  tensor_filter_subplugin *t = (tensor_filter_subplugin *) ptr;
  if (!t || t->sanity != _SANITY_CHECK || t->fwdesc.v1.subplugin_data != nullptr) {
    throw std::invalid_argument ("tfsp pointer is invalid");
  }

  return t;
}

/**
 * @brief C tensor-filter wrapper callback function, "close"
 */
void tensor_filter_subplugin::cpp_close (const GstTensorFilterProperties * prop,
    void **private_data)
{
  tensor_filter_subplugin *obj;

  try {
    obj = get_tfsp_with_checks (private_data);
  } catch (...) {
    /** @todo Write exception handlers. */
    return;
  }

  *private_data = nullptr;
  delete obj;
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "invoke"
 */
int tensor_filter_subplugin::cpp_invoke (const GstTensorFilterFramework * tf,
    const GstTensorFilterProperties *prop, void *private_data,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  tensor_filter_subplugin *obj;

  GET_TFSP_WITH_CHECKS (obj, private_data);

  try {
    obj->invoke (input, output);
  } catch (const std::exception & e) {
    /** @todo Write exception handlers. */

    return -EINVAL;
    /** @todo return different error codes according to exceptions */
  }
  return 0;
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "getFrameworkInfo"
 */
int tensor_filter_subplugin::cpp_getFrameworkInfo (
    const GstTensorFilterFramework * tf,
    const GstTensorFilterProperties * prop, void *private_data,
    GstTensorFilterFrameworkInfo *fw_info)
{
  tensor_filter_subplugin *obj;

  if (private_data == nullptr) {
    /** generate an emptyInstance and make query to it */
    const GstTensorFilterFramework * tfsp = tf;

    if (tfsp == nullptr)
      tfsp = nnstreamer_filter_find (prop->fwname);

    assert (tfsp);
    assert (tfsp->version == GST_TENSOR_FILTER_FRAMEWORK_V1);

    obj = (tensor_filter_subplugin *) tfsp->v1.subplugin_data;
  } else {
    GET_TFSP_WITH_CHECKS (obj, private_data);
  }

  try {
    obj->getFrameworkInfo (*fw_info);
  } catch (const std::exception & e) {
    /** @todo Write exception handlers. */

    return -EINVAL;
    /** @todo return different error codes according to exceptions */
  }
  return 0;
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "getModelInfo"
 */
int tensor_filter_subplugin::cpp_getModelInfo (
    const GstTensorFilterFramework * tf,
    const GstTensorFilterProperties * prop, void *private_data,
    model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  tensor_filter_subplugin *obj;

  GET_TFSP_WITH_CHECKS (obj, private_data);
  return obj->getModelInfo (ops, *in_info, *out_info);
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "eventHandler"
 */
int tensor_filter_subplugin::cpp_eventHandler (
    const GstTensorFilterFramework * tf,
    const GstTensorFilterProperties * prop, void *private_data, event_ops ops,
    GstTensorFilterFrameworkEventData *data)
{
  tensor_filter_subplugin *obj;

  GET_TFSP_WITH_CHECKS (obj, private_data);
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
  return -ENOENT;
}

} /* namespace nnstreamer */
