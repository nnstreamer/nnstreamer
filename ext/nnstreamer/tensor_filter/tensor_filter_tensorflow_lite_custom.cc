/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter for tensorflow-lite custom binary
 * Copyright (C) 2022 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file        tensor_filter_tensorflow_lite_custom.cc
 * @date        13 April 2022
 * @brief       Tensorflow-lite custom binary support for tensor_filter
 * @see         https://nnstreamer.github.io
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @todo        Delegation not supported. Add it when it's needed!
 *              (refer to the conventional tfl subplugin)
 * @todo        Model-change not supported. Add it when it's needed!
 *              (refer to the conventional tfl subplugin)
 */

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>

/** Supports tensorflow2-lite >= 2.4 only! */
#include <tensorflow2/lite/kernels/register.h>
#include <tensorflow2/lite/model.h>

namespace nnstreamer
{
namespace tensorfilter_tflite_cusom
{
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void _init_filter_tflc (void) __attribute__((constructor));
void _fini_filter_tflc (void) __attribute__((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief dlsym-ed methods of tensorflow-lite.
 * @todo Fill them in and fix the type info.
 */
typedef struct
{
  void * tflite__FlatBufferModel__BuildFromFile;
      /* tflite::FlatBufferModel::BuildFromFile (model_path); */
  void * tflite__InterpreterBuilder;
      /* tflite::InterpreterBuilder (*model, resolver) (&interpreter); */

  /** @todo For delegation
  tflite::Interpreter::TfLiteDelegatePtr (delegate, deleter);
  delegate = new tflite::StatefulNnApiDelegate (); // may have constructor-dlopen issue.
  */

  /** Classes
   * std::unique_ptr<tflite::Interpreter> interpreter;
   * std::unique_ptr<tflite::FlatBufferModel> model;
   * tflite::Interpreter::TfLiteDelegatePtr delegate_ptr; (delegate)
   */
} tfl_methods;

const char *symbols_to_load[] = {
};

/**
 * @brief Tensor-filter subplugin concrete class for tensorflow-lite.
 */
class tflc_subplugin final : public tensor_filter_subplugin
{ /* tflc = tensorflow-lite-custom */
  private:
  void *tfl; /**< dlopen instance */

  public:
  static void init_filter_tflc ();
  static void fini_filter_tflc ();

  tflc_subplugin ();
  ~tflc_subplugin ();

  /** implementing virtual methods of parent */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};


/** @brief tflc virtual method implementation */
tflc_subplugin::tflc_subplugin ()
{
}

/** @brief tflc virtual method implementation */
tflc_subplugin::~tflc_subplugin ()
{
}

/** @brief tflc virtual method implementation */
tensor_filter_subplugin &
tflc_subplugin::getEmptyInstance ()
{
  return *(new tflc_subplugin ());
}

/** @brief tflc virtual method implementation */
void
tflc_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  UNUSED (prop);
}

/** @brief tflc virtual method implementation */
void
tflc_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (input);
  UNUSED (output);
}

/** @brief tflc virtual method implementation */
void
tflc_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  UNUSED (info);
}

/** @brief tflc virtual method implementation */
int
tflc_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  UNUSED (ops);
  UNUSED (in_info);
  UNUSED (out_info);
  return 0;
}

/** @brief tflc virtual method implementation */
int
tflc_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return 0;
}


/** @brief tflc virtual method implementation */
void
tflc_subplugin::init_filter_tflc (void)
{
}

/** @brief tflc virtual method implementation */
void
tflc_subplugin::fini_filter_tflc (void)
{
}

/**
 * @brief initializer
 */
void
_init_filter_tflc ()
{
  tflc_subplugin::init_filter_tflc ();
}

/**
 * @brief finalizer
 */
void
_fini_filter_tflc ()
{
  tflc_subplugin::fini_filter_tflc ();
}
} /* namespace nnstreamer::tensorfilter_tflite_custom */
} /* namespace nnstreamer */
