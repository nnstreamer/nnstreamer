/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for DeepView RT
 * Copyright 2022 NXP
 */
/**
 * @file    tensor_filter_deepview_rt.cc
 * @date    7 Jun 2022
 * @brief   NNStreamer tensor-filter sub-plugin for DeepView RT engine
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Julien Vuillaumier <julien.vuillaumier@nxp.com>
 * @bug     No known bugs except for NYI items
 */
#include <fstream>
#include <limits.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>


#define DVRT_SUBPLUGIN_NAME "deepview-rt"


/**
 * @brief Glib logging domain
 *        set environment variable G_MESSAGES_DEBUG=deepview-rt for debug/info logs
 */
#define G_LOG_DOMAIN DVRT_SUBPLUGIN_NAME

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

#include <deepview_rt.h>

using namespace std;


namespace nnstreamer
{
namespace tensorfilter_dvrt
{

extern "C" {
void _init_filter_dvrt (void) __attribute__ ((constructor));
void _fini_filter_dvrt (void) __attribute__ ((destructor));
}

/**
 * @brief Default context cache size in MiB
 */
#define DVRT_CONTEXT_CACHE_SIZE_MB_DEFAULT 8

/**
 * @brief Default context mempool size in MiB
 */
#define DVRT_CONTEXT_MEMPOOL_SIZE_MB_DEFAULT 16


/**
 * @brief tensor_filter_subplugin concrete class for DeepViewRT.
 */
class dvrt_subplugin final : public tensor_filter_subplugin
{
  private:
  struct dvrt_options_s;

  GstTensorsInfo inputInfo; /**< The tensor info of input tensors */
  GstTensorsInfo outputInfo; /**< The tensor info of output tensors */
  GMappedFile *modelMap; /**< Model file mmaped to memory */

  NNContext *context; /**< Context for model load and runtime */
  NNEngine *engine; /**< Engine path for context acceleration */
  const NNModel *model; /**< Model blob pointer */
  vector<NNTensor *> inputTensors; /**< Input tensor from model */
  vector<NNTensor *> outputTensors; /**< Output tensor from model */

  static const char *name;
  static dvrt_subplugin *registeredRepresentation;

  void cleanup ();
  int parseCustomOptions (const GstTensorFilterProperties *props,
                          struct dvrt_options_s *opts);
  int initContext (dvrt_options_s *options);
  int initTensorsMeta ();
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int setTensorProp (gint isInput);
  int getTensorDim (gsize index, tensor_dim dim);
  int getTensorType (gsize index, tensor_type *type);

  public:
  static void init_filter_dvrt ();
  static void fini_filter_dvrt ();

  dvrt_subplugin ();
  ~dvrt_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info,
                                        GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *dvrt_subplugin::name = DVRT_SUBPLUGIN_NAME;

/**
 * @brief Options for dvrt context.
 */
struct dvrt_subplugin::dvrt_options_s {
  const gchar *modelPath; /**< rtm model path */
  const gchar *enginePath; /**< Engine path for context acceleration */
  guint cacheMb; /**< Context cache size in MiB */
  guint memPoolMb; /**< Context mempool size in MiB */
};

/**
 * @brief dvrt_subplugin class constructor.
 */
dvrt_subplugin::dvrt_subplugin ()
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));

  modelMap = nullptr;
  context  = nullptr;
  engine   = nullptr;
}

/** @brief cleanup resources used by dvrt subplugin */
void
dvrt_subplugin::cleanup ()
{
  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  if (context) {
    nn_context_model_unload (context);
    nn_context_release (context);
    context = nullptr;
  }

  if (engine) {
    nn_engine_unload (engine);
    g_free (engine);
    engine = nullptr;
  }

  if (modelMap) {
    g_mapped_file_unref (modelMap);
    modelMap = nullptr;
  }
}

/**
 * @brief dvrt_subplugin class destructor.
 */
dvrt_subplugin::~dvrt_subplugin ()
{
  cleanup ();
}

/** @brief get empty instance of deepviewrt subplugin */
tensor_filter_subplugin &
dvrt_subplugin::getEmptyInstance ()
{
  return *(new dvrt_subplugin ());
}


/**
 * @brief Internal function to get the option for dvrt context.
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::parseCustomOptions (const GstTensorFilterProperties *props,
                                    struct dvrt_options_s *opts)
{
  if (props->num_models == 1)
    opts->modelPath = props->model_files[0];
  else
    opts->modelPath = nullptr;

  opts->enginePath = nullptr;
  opts->cacheMb    = DVRT_CONTEXT_CACHE_SIZE_MB_DEFAULT;
  opts->memPoolMb  = DVRT_CONTEXT_MEMPOOL_SIZE_MB_DEFAULT;

  if (props->custom_properties) {
    gchar **strv;
    guint i, len;

    strv = g_strsplit (props->custom_properties, ",", -1);
    len = g_strv_length (strv);

    for (i = 0; i < len; ++i) {
      gchar **pair = g_strsplit (strv[i], ":", -1);

      if (g_strv_length (pair) > 1) {
        g_strstrip (pair[0]);
        g_strstrip (pair[1]);

        if (g_ascii_strcasecmp (pair[0], "Engine") == 0)
          opts->enginePath = g_strdup (pair[1]);
        else if (g_ascii_strcasecmp (pair[0], "Cache") == 0)
          opts->cacheMb = g_ascii_strtoll (pair[1], nullptr, 10);
        else if (g_ascii_strcasecmp (pair[0], "MemPool") == 0)
          opts->memPoolMb = g_ascii_strtoll (pair[1], nullptr, 10);
      }

      g_strfreev (pair);
    }

    g_strfreev (strv);
  }

  return 0;
}

/**
 * @brief return the type of tensor
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::getTensorType (gsize index, tensor_type *type)
{
  NNTensorType _type;
  tensor_type res;

  *type = _NNS_END;

  _type = nn_model_layer_datatype_id (model, index);
  switch (_type) {
  case NNTensorType_I8:
    res = _NNS_INT8;
    break;
  case NNTensorType_U8:
    res = _NNS_UINT8;
    break;
  case NNTensorType_I16:
    res = _NNS_INT16;
    break;
  case NNTensorType_U16:
    res = _NNS_UINT16;
    break;
  case NNTensorType_I32:
    res = _NNS_INT32;
    break;
  case NNTensorType_U32:
    res = _NNS_UINT32;
    break;
  case NNTensorType_I64:
    res = _NNS_INT64;
    break;
  case NNTensorType_U64:
    res = _NNS_UINT64;
    break;
  case NNTensorType_F32:
    res = _NNS_FLOAT32;
    break;
  case NNTensorType_F64:
    res = _NNS_FLOAT64;
    break;
  case NNTensorType_F16:
  default:
    nns_logw ("Tensor type not supported: %d", (gint)_type);
    return -EINVAL;
  }

  *type = res;
  return 0;
}

/**
 * @brief return the shape of tensor
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::getTensorDim (gsize index, tensor_dim dim)
{
  gsize dims;
  const gint32 *shape;
  gsize i;

  shape = nn_model_layer_shape (model, index, &dims);
  if (dims > NNS_TENSOR_RANK_LIMIT) {
    nns_logw ("Shape rank too high: %zu max: %d", dims, NNS_TENSOR_RANK_LIMIT);
    return -EINVAL;
  }

  /* the order of dimension is reversed at CAPS negotiation */
  for (i = 0; i < dims; i++)
    dim[dims - i - 1] = shape[i];

  /* fill remaining entries with 1 */
  for (i = dims; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief fetch and setup input/ouput tensors metadata
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::setTensorProp (gint isInput)
{
  GstTensorsInfo *tensorMeta;
  const guint32 *indices;
  gsize num;
  vector<NNTensor *> *tensors;
  const gchar *tag = isInput ? "input" : "output";

  if (isInput) {
    tensorMeta = std::addressof (inputInfo);
    indices = nn_model_inputs (model, &num);
    tensors = &inputTensors;
  } else {
    tensorMeta = std::addressof (outputInfo);
    indices = nn_model_outputs (model, &num);
    tensors = &outputTensors;
  }

  if (num > NNS_TENSOR_SIZE_LIMIT) {
    nns_logw ("Too many %s tensors: %zu max: %d",
    tag, num, NNS_TENSOR_SIZE_LIMIT);
    return -EINVAL;
  }
  tensorMeta->num_tensors = num;
  tensors->clear ();
  tensors->reserve (num);

  for (size_t i = 0; i < num; i++) {
    gsize index = indices[i];
    NNTensor *tensor = nn_context_tensor_index (context, index);
    tensors->push_back (tensor);

    const gchar *name = nn_model_layer_name (model, index);
    tensorMeta->info[i].name = g_strdup (name);
    if (getTensorDim (index, tensorMeta->info[i].dimension))
      return -EINVAL;

    if (getTensorType (index, &tensorMeta->info[i].type))
      return -EINVAL;

    gchar *dim;
    dim = gst_tensor_get_dimension_string (tensorMeta->info[i].dimension);
    ml_logd ("tensorMeta[%zu] >> name[%s], type[%d], dim[%s]", i,
             tensorMeta->info[i].name, tensorMeta->info[i].type, dim);
    g_free (dim);
  }

  return 0;
}

/**
 * @brief fetch and setup ouput tensors metadata
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::setOutputTensorProp ()
{
  return setTensorProp (FALSE);
}

/**
 * @brief fetch and setup input tensors metadata
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::setInputTensorProp ()
{
  return setTensorProp (TRUE);
}

/**
 * @brief create context for rtm model.
 * @return 0 if OK. non-zero if error.
 */
int
dvrt_subplugin::initContext (dvrt_options_s *options)
{
  GError *err = nullptr;
  int nerr;
  NNError nnerror;

  if (!options->modelPath)
    return -ENOENT;

  GMappedFile *modelMap = g_mapped_file_new (options->modelPath, FALSE, &err);
  if (!modelMap || err) {
    nns_logw ("Could not map model file %s %s",
              options->modelPath, err->message);
    g_clear_error (&err);
    return -ENOENT;
  }

  gsize size = g_mapped_file_get_length (modelMap);
  model = g_mapped_file_get_contents (modelMap);
  nerr = nn_model_validate (model, size);
  if (nerr) {
    nns_logw ("Model validation failed %s", nn_model_validate_error (nerr));
    return -EINVAL;
  }

  if (options->enginePath) {
    NNEngine *e = (NNEngine *) g_malloc (nn_engine_sizeof ());
    engine = nn_engine_init (e);
    nnerror = nn_engine_load (engine, options->enginePath);
    if (nnerror) {
      nns_logw ("Engine load failed %s %s",
                options->enginePath, nn_strerror (nnerror));
      return -ENOENT;
    }
  }

  context = nn_context_init (engine,
            (size_t) (options->memPoolMb * 1024UL * 1024UL),
            nullptr,
            (size_t) (options->cacheMb * 1024UL * 1024UL),
            nullptr);
  if (!context) {
    nns_logw ("Context init failed");
    return -ENOMEM;
  }

  nnerror = nn_context_model_load (context, size, model);
  if (nnerror) {
    nns_logw ("Context model load failed %s", nn_strerror (nnerror));
    return -EINVAL;
  }

  return 0;
}

/** @brief configure dvrt instance */
void
dvrt_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  struct dvrt_options_s options = {};
  int ret;

  ret = parseCustomOptions (prop, &options);
  if (ret)
    goto done;

  ret = initContext (&options);
  if (ret)
    goto done;

  ret = setInputTensorProp ();
  if (ret)
    goto done;

  ret = setOutputTensorProp ();

done:
  g_free ((gpointer) options.enginePath);
  if (ret) {
    cleanup ();
    throw std::invalid_argument ("Instance configuration failed");
  }
}

/** @brief invoke using dvrt */
void
dvrt_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  g_assert (inputTensors.size () == inputInfo.num_tensors);
  for (gsize i = 0; i < inputInfo.num_tensors; ++i) {
    NNTensor *tensor = inputTensors[i];
    g_assert (tensor);
    gsize size = nn_tensor_size (tensor);
    g_assert (size == input[i].size);

    void *data = nn_tensor_mapwo (tensor);
    g_assert (data);
    memcpy (data, input[i].data, input[i].size);
    nn_tensor_unmap (tensor);

    nns_logd ("Invoke Input copy to (%p) (%zu) bytes", data, input[i].size);
  }

  NNError err = nn_context_run (context);
  if (err) {
    nns_logw ("Context run failed %s", nn_strerror (err));
    throw std::runtime_error ("Invoking DeepView RT failed.");
  }

  g_assert (outputTensors.size () == outputInfo.num_tensors);
  for (gsize i = 0; i < outputInfo.num_tensors; ++i) {
    NNTensor *tensor = outputTensors[i];
    g_assert (tensor);
    gsize size = nn_tensor_size (tensor);
    g_assert (size == output[i].size);

    const void *data = nn_tensor_mapro (tensor);
    g_assert (data);
    memcpy (output[i].data, data, output[i].size);
    nn_tensor_unmap (tensor);

    nns_logd ("Invoke Output copy from (%p) (%zu) bytes",
              data, output[i].size);
  }
}

/** @brief Get framework information */
void
dvrt_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Method to get the model information.
 */
int
dvrt_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
dvrt_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

dvrt_subplugin *dvrt_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
dvrt_subplugin::init_filter_dvrt (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<dvrt_subplugin> ();
  nnstreamer_filter_set_custom_property_desc (name,
      "Cache",   "Context cache size in MiB",
      "MemPool", "Context mempool size in MiB",
      "Engine",  "Engine plugin path for context acceleration",
      nullptr);
}

/** @brief initializer */
void
_init_filter_dvrt ()
{
  dvrt_subplugin::init_filter_dvrt ();
}

/** @brief Destruct the subplugin */
void
dvrt_subplugin::fini_filter_dvrt (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief finalizer */
void
_fini_filter_dvrt ()
{
  dvrt_subplugin::fini_filter_dvrt ();
}

}  /* namespace nnstreamer::tensorfilter_dvrt */
} /* namespace nnstreamer */
