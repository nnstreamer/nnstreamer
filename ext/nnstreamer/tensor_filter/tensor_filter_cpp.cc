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
#include <iostream>
#include <string>
#include <assert.h>

#include <errno.h>
#include <string.h>
#include <glib.h>
#include <gmodule.h>

#include <nnstreamer_plugin_api_filter.h>

#include "tensor_filter_cpp.hh"

std::unordered_map<std::string, tensor_filter_cpp*> tensor_filter_cpp::filters;
std::vector<void *> tensor_filter_cpp::handles;
G_LOCK_DEFINE_STATIC (lock_handles);

static gchar filter_subplugin_cpp[] = "cpp";
bool tensor_filter_cpp::close_all_called = false;

static GstTensorFilterFramework NNS_support_cpp = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = tensor_filter_cpp::open,
  .close = tensor_filter_cpp::close,
};

G_BEGIN_DECLS
void init_filter_cpp (void) __attribute__ ((constructor));
void fini_filter_cpp (void) __attribute__ ((destructor));

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_cpp (void)
{
  NNS_support_cpp.name = filter_subplugin_cpp;
  NNS_support_cpp.allow_in_place = FALSE;      /** @todo: support this to optimize performance later. */
  NNS_support_cpp.allocate_in_invoke = FALSE;
  NNS_support_cpp.run_without_model = FALSE;
  NNS_support_cpp.verify_model_path = FALSE;
  NNS_support_cpp.invoke_NN = tensor_filter_cpp::invoke;
  NNS_support_cpp.getInputDimension = tensor_filter_cpp::getInputDim;
  NNS_support_cpp.getOutputDimension = tensor_filter_cpp::getOutputDim;
  NNS_support_cpp.setInputDimension = tensor_filter_cpp::setInputDim;
  nnstreamer_filter_probe (&NNS_support_cpp);
}

/** @brief Destruct the subplugin */
void
fini_filter_cpp (void)
{
  nnstreamer_filter_exit (NNS_support_cpp.name);
  tensor_filter_cpp::close_all_handles ();
}
G_END_DECLS

#define loadClass(name, ptr) \
  class tensor_filter_cpp *name = (tensor_filter_cpp *) *(ptr); \
  assert (false == close_all_called); \
  assert (*(ptr)); \
  assert (name->isValid());

/**
 * @brief Class constructor
 */
tensor_filter_cpp::tensor_filter_cpp(const char *name): validity(0xdeafdead), name(g_strdup(name)), ref_count(0), prop(NULL)
{
}

/**
 * @brief Class destructor
 */
tensor_filter_cpp::~tensor_filter_cpp()
{
  g_free((gpointer) name);
}

/**
 * @brief Check if the given object (this) is valid.
 */
bool tensor_filter_cpp::isValid()
{
  return this->validity == 0xdeafdead;
}

/**
 * @brief Register the c++ filter
 */
int tensor_filter_cpp::__register (
    class tensor_filter_cpp *filter, unsigned int ref_count)
{
  if (filters.find (filter->name) != filters.end())
    return -EINVAL; /** Already registered */
  if (ref_count)
    filter->ref_count = ref_count;
  filters[filter->name] = filter;

  return 0;
}

/**
 * @brief Unregister the c++ filter from unordered map
 */
int tensor_filter_cpp::__unregister (const char *name)
{
  if (filters.find (name) == filters.end())
    return -EINVAL; /** Not found */
  if (filters[name]->ref_count > 0) {
    unsigned int cnt = filters[name]->ref_count;
    g_critical ("The reference counter of c++ filter, %s, is %u. Anyway, we are closing this because this is being closed by destructor of .so file.", name, cnt);
  }
  size_t num = filters.erase (name);
  if (num != 1)
    return -EINVAL; /** Cannot erase */

  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
int tensor_filter_cpp::getInputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info)
{
  loadClass(cpp, private_data);
  return cpp->getInputDim (info);
}

/**
 * @brief Standard tensor_filter callback
 */
int tensor_filter_cpp::getOutputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info)
{
  loadClass(cpp, private_data);
  return cpp->getOutputDim (info);
}

/**
 * @brief Standard tensor_filter callback
 */
int tensor_filter_cpp::setInputDim (const GstTensorFilterProperties *prop, void **private_data, const GstTensorsInfo *in, GstTensorsInfo *out)
{
  loadClass(cpp, private_data);
  return cpp->setInputDim (in, out);
}

/**
 * @brief Standard tensor_filter callback
 */
int tensor_filter_cpp::invoke (const GstTensorFilterProperties *prop, void **private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  loadClass(cpp, private_data);
  return cpp->invoke(input, output);
}

/**
 * @brief Printout only once for a given error
 */
__attribute__((format(printf, 3, 4)))
static void g_printerr_once (const char *file, int line, const char *fmt,
    ...)
{
  static guint file_hash = 0;
  static int _line = 0;

  if (file_hash != g_str_hash(file) || _line != line) {
    char buffer[256];
    file_hash = g_str_hash(file);
    _line = line;

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, 256, fmt, args);
    g_printerr("%s", buffer);
    va_end(args);
  }
}

/**
 * @brief Standard tensor_filter callback
 */
int tensor_filter_cpp::open (const GstTensorFilterProperties *prop, void **private_data)
{
  class tensor_filter_cpp *cpp;

  if (*private_data) {
    /** Reloading. Unload it to reload */
    close (prop, private_data);
    *private_data = NULL;
  }

  if (filters.find(prop->model_files[0]) == filters.end()) {
    /* model_files may be really path to .so file. try to open it */
    if (prop->num_models < 2)
      return -EINVAL;

    GModule *module = g_module_open (prop->model_files[1], (GModuleFlags) 0);
    if (!module) {
      g_printerr_once (__FILE__, __LINE__, "C++ custom filter %s cannot be found: opening %s failed\n", prop->model_files[0], prop->model_files[1]);
      return -EINVAL; /** Model file / name not found */
    }

    if (filters.find(prop->model_files[0]) == filters.end()) {
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
  cpp->prop = prop;

  NNS_support_cpp.allocate_in_invoke = ! cpp->isAllocatedBeforeInvoke();

  return 0;
}

/**
 * @brief Standard tensor_filter callback
 */
void tensor_filter_cpp::close (const GstTensorFilterProperties *prop, void **private_data)
{
  loadClass(cpp, private_data);

  g_assert (cpp->ref_count > 0);
  /** The class is deallocated from unordered_map if ref_count hits 0 */
  cpp->ref_count--;
}

/**
 * @brief Call dlclose for all handle
 */
void tensor_filter_cpp::close_all_handles ()
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
