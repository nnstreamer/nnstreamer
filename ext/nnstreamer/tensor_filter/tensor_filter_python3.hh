/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 */
/**
 * @file    tensor_filter_subplugin_python3.hh
 * @date    26 Mar 2024
 * @brief   NNStreamer tensor-filter subplugin python header
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Yelin Jeong <yelini.jeong@samsung.com>
 * @bug     No known bugs
 */

#ifndef __TENSOR_FILTER_SUBPLUGIN_PYTHON3_H__
#define __TENSOR_FILTER_SUBPLUGIN_PYTHON3_H__

/* nnstreamer plugin api headers */
#include <map>
#include <nnstreamer_conf.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "nnstreamer_python3_helper.h"

/** @brief Callback type for custom filter */
typedef enum _cb_type
{
  CB_SETDIM = 0,
  CB_GETDIM,

  CB_END,
} cb_type;

/**
 * @brief	Python embedding core structure
 */
class PYCore
{
public:
  /**
   * member functions.
   */
  PYCore (const char *_script_path, const char *_custom);
   ~PYCore ();

  int init (const GstTensorFilterProperties * prop);
  const char *getScriptPath ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int setInputTensorDim (const GstTensorsInfo * in_info,
      GstTensorsInfo * out_info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

  void freeOutputTensors (void *data);

  /** @brief Return callback type */
  cb_type getCbType ()
  {
    return callback_type;
  }

private:
  int loadScript ();
  const std::string script_path; /**< from model_path property */
  const std::string module_args; /**< from custom property */

  std::string module_name;
  std::map < void *, PyArrayObject * >outputArrayMap;

  cb_type callback_type;

  PyObject *core_obj;
  PyObject *shape_cls;

  GstTensorsInfo inputTensorMeta; /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta; /**< The tensor info of output tensors */

  bool configured; /**< True if the script is successfully loaded */
  void *handle; /**< returned handle by dlopen() */
  GMutex py_mutex;

  int checkTensorSize (GstTensorMemory * output, PyArrayObject * array);
  int checkTensorType (int nns_type, int np_type);

  /** @brief Lock python-related actions */
  void Py_LOCK ()
  {
    g_mutex_lock (&py_mutex);
  }
  /** @brief Unlock python-related actions */
  void Py_UNLOCK ()
  {
    g_mutex_unlock (&py_mutex);
  }
};

namespace nnstreamer
{

/**
 * @brief Class for Python3 subplugin
 */
  class TensorFilterPython:public tensor_filter_subplugin
  {
  public:
    TensorFilterPython ();

    /* mandatory methods */
    tensor_filter_subplugin & getEmptyInstance ();
    void configure_instance (const GstTensorFilterProperties * prop);
    void invoke (const GstTensorMemory * input, GstTensorMemory * output);
    void getFrameworkInfo (GstTensorFilterFrameworkInfo & info);
    int getModelInfo (model_info_ops ops, GstTensorsInfo & in_info,
        GstTensorsInfo & out_info);
    int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData & data);

    /* static methods */
    static void init_filter_py ();
    static void fini_filter_py ();

  private:
      PYCore * core;

    static TensorFilterPython *registered;
    static const char *name;
    static const accl_hw hw_list[];
    static const int num_hw;
  };
}                               /* namespace nnstreamer */

#endif                          /* __TENSOR_FILTER_SUBPLUGIN_PYTHON3_H__ */
