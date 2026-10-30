/**
 * GStreamer Tensor_Filter, Customized C++ Module
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_filter_cpp.cc
 * @date	26 Sep 2019
 * @brief	Tensor_filter subplugin for C++ custom filters.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @note This is experimental. The API and Class definition is NOT stable.
 *       If you want to write an OpenCV custom filter for nnstreamer, this is a good choice.
 *
 */
#include <assert.h>
#include <iostream>
#include <string>

#include <errno.h>
#include <glib.h>
#include <gmodule.h>
#include <string.h>

#include <nnstreamer_log.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include "tensor_filter_cpp.hh"

std::unordered_map<std::string, tensor_filter_cpp *> tensor_filter_cpp::filters;
std::vector<void *> tensor_filter_cpp::handles;
thread_local const GstTensorFilterProperties *tensor_filter_cpp::prop = nullptr;
G_LOCK_DEFINE_STATIC (lock_handles);

static gchar filter_subplugin_cpp[] = "cpp";
bool tensor_filter_cpp::close_all_called = false;

/**
 * @brief The v1 invoke callback, dispatching to tensor_filter_cpp::invoke
 */
static int
cpp_v1_invoke (const GstTensorFilterFramework *self, GstTensorFilterProperties *prop,
    void *private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (self);
  return tensor_filter_cpp::invoke (prop, &private_data, input, output);
}

/**
 * @brief The v1 getFrameworkInfo callback. An opened filter tells its own allocation contract.
 */
static int
cpp_v1_getFrameworkInfo (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    GstTensorFilterFrameworkInfo *fw_info)
{
  UNUSED (self);
  UNUSED (prop);

  fw_info->name = filter_subplugin_cpp;
  fw_info->allow_in_place = FALSE; /** @todo: support this to optimize performance later. */
  fw_info->allocate_in_invoke = FALSE;
  if (private_data)
    fw_info->allocate_in_invoke
        = (tensor_filter_cpp::allocateInInvoke (&private_data) == 0);
  fw_info->run_without_model = FALSE;
  fw_info->verify_model_path = FALSE;
  fw_info->hw_list = nullptr;
  fw_info->num_hw = 0;
  fw_info->accl_auto = ACCL_NONE;
  fw_info->accl_default = ACCL_NONE;
  fw_info->statistics = nullptr;

  return 0;
}

/**
 * @brief The v1 getModelInfo callback, dispatching to the dimension callbacks of tensor_filter_cpp
 * @note v0 asked for the input and output dimensions one by one, and
 *       tensor_filter derived the missing one itself. GET_IN_OUT_INFO asks
 *       for both, so a filter with only one of the two fixed fills the other
 *       here, the way tensor_filter did in v0: a fixed input gets its output
 *       from setInputDim (none with invoke-dynamic, where tensor_filter takes
 *       the output from invoke), and a fixed output is used when setInputDim
 *       fails.
 */
static int
cpp_v1_getModelInfo (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  GstTensorsInfo fixed_in, fixed_out;
  int ret;
  UNUSED (self);

  if (!private_data || !in_info || !out_info)
    return -EINVAL;

  if (ops == GET_IN_OUT_INFO) {
    ret = tensor_filter_cpp::getInputDim (prop, &private_data, in_info);
    if (ret == 0 && tensor_filter_cpp::getOutputDim (prop, &private_data, out_info) != 0) {
      gst_tensors_info_free (out_info);
      if (!prop || !prop->invoke_dynamic)
        ret = tensor_filter_cpp::setInputDim (prop, &private_data, in_info, out_info);
    }
    return ret;
  } else if (ops == SET_INPUT_INFO) {
    ret = tensor_filter_cpp::setInputDim (prop, &private_data, in_info, out_info);
    if (ret != 0) {
      gst_tensors_info_init (&fixed_in);
      gst_tensors_info_init (&fixed_out);
      if (tensor_filter_cpp::getInputDim (prop, &private_data, &fixed_in) != 0
          && tensor_filter_cpp::getOutputDim (prop, &private_data, &fixed_out) == 0) {
        gst_tensors_info_free (out_info);
        gst_tensors_info_copy (out_info, &fixed_out);
        ret = 0;
      }
      gst_tensors_info_free (&fixed_in);
      gst_tensors_info_free (&fixed_out);
    }
    return ret;
  }

  return -ENOENT;
}

/**
 * @brief The v1 eventHandler callback. No event is handled.
 */
static int
cpp_v1_eventHandler (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data, event_ops ops,
    GstTensorFilterFrameworkEventData *data)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  UNUSED (ops);
  UNUSED (data);

  return -ENOENT;
}

static GstTensorFilterFramework NNS_support_cpp = { .version = GST_TENSOR_FILTER_FRAMEWORK_V1,
  .open = tensor_filter_cpp::open,
  .close = tensor_filter_cpp::close,
  { .v1 = {
        .invoke = cpp_v1_invoke,
        .getFrameworkInfo = cpp_v1_getFrameworkInfo,
        .getModelInfo = cpp_v1_getModelInfo,
        .eventHandler = cpp_v1_eventHandler,
        .subplugin_data = nullptr,
    } } };

G_BEGIN_DECLS
void init_filter_cpp (void) __attribute__ ((constructor));
void fini_filter_cpp (void) __attribute__ ((destructor));

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_cpp (void)
{
  nnstreamer_filter_probe (&NNS_support_cpp);
}

/** @brief Destruct the subplugin */
void
fini_filter_cpp (void)
{
  nnstreamer_filter_exit (filter_subplugin_cpp);
  tensor_filter_cpp::close_all_handles ();
}
G_END_DECLS

#define loadClass(name, ptr)                                    \
  class tensor_filter_cpp *name = (tensor_filter_cpp *) *(ptr); \
  assert (false == close_all_called);                           \
  assert (*(ptr));                                              \
  assert (name->isValid ());

/**
 * @brief Keeps prop set to the calling element's properties while it lives
 */
class tensor_filter_cpp::prop_scope
{
  public:
  /** @brief Expose the caller's properties on this thread */
  prop_scope (const GstTensorFilterProperties *caller)
      : outer (tensor_filter_cpp::prop)
  {
    tensor_filter_cpp::prop = caller;
  }

  /** @brief Restore what was exposed before */
  ~prop_scope ()
  {
    tensor_filter_cpp::prop = outer;
  }

  private:
  const GstTensorFilterProperties *outer; /**< prop when this scope began */
};

/**
 * @brief Class constructor
 */
tensor_filter_cpp::tensor_filter_cpp (const char *name)
    : validity (0xdeafdead), name (g_strdup (name)), ref_count (0)
{
}

/**
 * @brief Class destructor
 */
tensor_filter_cpp::~tensor_filter_cpp ()
{
  g_free ((gpointer) name);
}

/**
 * @brief Check if the given object (this) is valid.
 */
bool
tensor_filter_cpp::isValid ()
{
  return this->validity == 0xdeafdead;
}

/**
 * @brief Register the c++ filter
 */
int
tensor_filter_cpp::__register (class tensor_filter_cpp *filter, unsigned int ref_count)
{
  if (filters.find (filter->name) != filters.end ())
    return -EINVAL; /** Already registered */
  if (ref_count)
    filter->ref_count = ref_count;
  filters[filter->name] = filter;

  return 0;
}

/**
 * @brief Unregister the c++ filter from unordered map
 */
int
tensor_filter_cpp::__unregister (const char *name)
{
  if (filters.find (name) == filters.end ())
    return -EINVAL; /** Not found */
  if (filters[name]->ref_count > 0) {
    unsigned int cnt = filters[name]->ref_count;
    ml_loge ("The reference counter of c++ filter, %s, is %u. Anyway, we are closing this because this is being closed by destructor of .so file.",
        name, cnt);
  }
  size_t num = filters.erase (name);
  if (num != 1)
    return -EINVAL; /** Cannot erase */

  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
int
tensor_filter_cpp::getInputDim (const GstTensorFilterProperties *prop,
    void **private_data, GstTensorsInfo *info)
{
  loadClass (cpp, private_data);
  prop_scope scope (prop);
  return cpp->getInputDim (info);
}

/**
 * @brief Standard tensor_filter callback
 */
int
tensor_filter_cpp::getOutputDim (const GstTensorFilterProperties *prop,
    void **private_data, GstTensorsInfo *info)
{
  loadClass (cpp, private_data);
  prop_scope scope (prop);
  return cpp->getOutputDim (info);
}

/**
 * @brief Standard tensor_filter callback
 */
int
tensor_filter_cpp::setInputDim (const GstTensorFilterProperties *prop,
    void **private_data, const GstTensorsInfo *in, GstTensorsInfo *out)
{
  loadClass (cpp, private_data);
  prop_scope scope (prop);
  return cpp->setInputDim (in, out);
}

/**
 * @brief Standard tensor_filter callback
 */
int
tensor_filter_cpp::invoke (const GstTensorFilterProperties *prop,
    void **private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  loadClass (cpp, private_data);
  prop_scope scope (prop);
  return cpp->invoke (input, output);
}

/**
 * @brief Standard tensor_filter callback
 */
int
tensor_filter_cpp::allocateInInvoke (void **private_data)
{
  loadClass (cpp, private_data);
  return cpp->isAllocatedBeforeInvoke () ? -EINVAL : 0;
}

/**
 * @brief Printout only once for a given error
 */
__attribute__ ((format (printf, 3, 4))) static void
g_printerr_once (const char *file, int line, const char *fmt, ...)
{
  static guint file_hash = 0;
  static int _line = 0;

  if (file_hash != g_str_hash (file) || _line != line) {
    char buffer[256];
    file_hash = g_str_hash (file);
    _line = line;

    va_list args;
    va_start (args, fmt);
    vsnprintf (buffer, 256, fmt, args);
    g_printerr ("%s", buffer);
    va_end (args);
  }
}

/**
 * @brief Standard tensor_filter callback
 */
int
tensor_filter_cpp::open (const GstTensorFilterProperties *prop, void **private_data)
{
  class tensor_filter_cpp *cpp;

  if (*private_data) {
    /** Reloading. Unload it to reload */
    close (prop, private_data);
    *private_data = NULL;
  }

  if (filters.find (prop->model_files[0]) == filters.end ()) {
    /* model_files may be really path to .so file. try to open it */
    if (prop->num_models < 2)
      return -EINVAL;

    GModule *module = g_module_open (prop->model_files[1], (GModuleFlags) 0);
    if (!module) {
      g_printerr_once (__FILE__, __LINE__,
          "C++ custom filter %s cannot be found: opening %s failed\n",
          prop->model_files[0], prop->model_files[1]);
      return -EINVAL; /** Model file / name not found */
    }

    if (filters.find (prop->model_files[0]) == filters.end ()) {
      /** It's still not found. it's not there. */
      g_module_close (module);
      g_printerr_once (__FILE__, __LINE__, "C++ custom filter %s is not found in %s.\n",
          prop->model_files[0], prop->model_files[1]);
      return -EINVAL;
    }

    /** We do not know until when this handle might be required: user may
     * invoke functions from it at anytime while the pipeline is not
     * closed */
    G_LOCK (lock_handles);
    handles.push_back ((void *) module);
    G_UNLOCK (lock_handles);
  }

  *private_data = cpp = filters[prop->model_files[0]];
  cpp->ref_count++;

  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
void
tensor_filter_cpp::close (const GstTensorFilterProperties *prop, void **private_data)
{
  loadClass (cpp, private_data);
  UNUSED (prop);

  g_assert (cpp->ref_count > 0);
  /** The class is deallocated from unordered_map if ref_count hits 0 */
  cpp->ref_count--;
}

/**
 * @brief Call dlclose for all handle
 */
void
tensor_filter_cpp::close_all_handles ()
{
  assert (false == close_all_called);
/**
 * Ubuntu 16.04 / GLIBC 2.23 Workaround
 * If we do dlclose at exit() function, it may incur
 * https://bugzilla.redhat.com/show_bug.cgi?id=1264556#c42
 * , which is a GLIBC bug at 2.23.
 * The corresponding error message is:
 * Inconsistency detected by ld.so: dl-close.c: 811:
 * _dl_close: Assertion `map->l_init_called' failed!
 */
#if defined(__GLIBC__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ <= 23)
/* Do not call dlclose */
#else
  G_LOCK (lock_handles);
  for (void *handle : handles) {
    g_module_close ((GModule *) handle);
  }
  G_UNLOCK (lock_handles);
#endif
  close_all_called = true;
}
