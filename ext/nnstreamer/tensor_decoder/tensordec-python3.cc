/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "python3"
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file        tensordec-python3.cc
 * @date        May 06 2021
 * @brief       NNStreamer tensor-decoder subplugin, "python3",
 *              which decodes tensor or tensors to any media type.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs except for NYI items
 */

#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_util.h>
#include "tensordecutil.h"
#include "nnstreamer_python3_helper.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_decoder_py (void) __attribute__ ((constructor));
void fini_decoder_py (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief	Python embedding core structure
 */
class PYDecoderCore
{
  public:
  /**
   * member functions.
   */
  PYDecoderCore (const char *_script_path);
  ~PYDecoderCore ();

  int init ();
  const char *getScriptPath ();
  GstFlowReturn decode (const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf);
  GstCaps *getOutCaps (const GstTensorsConfig *config);

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

  private:
  std::string module_name;
  const std::string script_path;
  PyObject *shape_cls;
  PyObject *core_obj;
  void *handle; /**< returned handle by dlopen() */
  GMutex py_mutex;
};

/**
 * @brief	PYDecoderCore creator
 * @param	_script_path	: the logical path to '{script_name}.py' file
 * @note	the script of _script_path will be loaded simultaneously
 * @return	Nothing
 */
PYDecoderCore::PYDecoderCore (const char *_script_path)
    : script_path (_script_path)
{
  if (openPythonLib (&handle))
    throw std::runtime_error (dlerror ());

  _import_array (); /** for numpy */

  /**
   * Parse script path to get module name
   * The module name should drop its extension (i.e., .py)
   */
  module_name = script_path;
  const size_t last_idx = module_name.find_last_of ("/\\");

  if (last_idx != std::string::npos)
    module_name.erase (0, last_idx + 1);

  const size_t ext_idx = module_name.rfind ('.');
  if (ext_idx != std::string::npos)
    module_name.erase (ext_idx);

  addToSysPath (script_path.substr (0, last_idx).c_str ());

  core_obj = NULL;
  shape_cls = NULL;

  g_mutex_init (&py_mutex);
}

/**
 * @brief	PYDecoderCore Destructor
 * @return	Nothing
 */
PYDecoderCore::~PYDecoderCore ()
{
  Py_SAFEDECREF (core_obj);
  Py_SAFEDECREF (shape_cls);
  PyErr_Clear ();

  dlclose (handle);
  g_mutex_clear (&py_mutex);
}

/**
 * @brief	decode tensor(s) to any media stream
 */
GstFlowReturn
PYDecoderCore::decode (const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  gboolean need_alloc;
  size_t mem_size;
  int rate_n = 0, rate_d = 1;
  PyObject *output = NULL;
  PyObject *raw_data, *in_info;
  GstFlowReturn ret = GST_FLOW_OK;

  Py_LOCK ();
  raw_data = PyList_New (config->info.num_tensors);
  in_info = PyList_New (config->info.num_tensors);
  rate_n = config->rate_n;
  rate_d = config->rate_d;

  for (unsigned int i = 0; i < config->info.num_tensors; i++) {
    tensor_type nns_type = config->info.info[i].type;
    npy_intp input_dims[] = { (npy_intp) (input[i].size / gst_tensor_get_element_size (nns_type)) };
    PyObject *input_array = PyArray_SimpleNewFromData (
        1, input_dims, getNumpyType (nns_type), input[i].data);
    PyList_SetItem (raw_data, i, input_array);

    PyObject *shape = PyTensorShape_New (shape_cls, &config->info.info[i]);
    PyList_SetItem (in_info, i, shape);
  }

  if (!PyObject_HasAttrString (core_obj, (char *)"decode")) {
    Py_ERRMSG ("Cannot find 'decode'");
    ret = GST_FLOW_ERROR;
    goto done;
  }

  output = PyObject_CallMethod (core_obj, "decode",  "OOii", raw_data, in_info, rate_n, rate_d);

  if (output) {
    need_alloc = (gst_buffer_get_size (outbuf) == 0);
    mem_size = PyBytes_Size (output);

    if (need_alloc) {
      out_mem = gst_allocator_alloc (NULL, mem_size, NULL);
    } else {
      if (gst_buffer_get_size (outbuf) < mem_size) {
        gst_buffer_set_size (outbuf, mem_size);
      }
      out_mem = gst_buffer_get_all_memory (outbuf);
    }

    if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
      gst_memory_unref (out_mem);
      nns_loge ("Cannot map gst memory (tensor decoder flexbuf)\n");
      ret = GST_FLOW_ERROR;
      goto done;
    }

    memcpy (out_info.data, PyBytes_AsString (output), mem_size);

    gst_memory_unmap (out_mem, &out_info);

    if (need_alloc)
      gst_buffer_append_memory (outbuf, out_mem);
    else
      gst_memory_unref (out_mem);

    Py_SAFEDECREF (output);
  } else {
    Py_ERRMSG ("Fail to get output from 'convert'");
    ret = GST_FLOW_ERROR;
  }

done:
  Py_UNLOCK ();
  return ret;
}

/**
 * @brief	get output caps
 */
GstCaps *
PYDecoderCore::getOutCaps (const GstTensorsConfig *config)
{
  PyObject *result = NULL;
  GstCaps *caps = NULL;
  UNUSED (config);

  Py_LOCK ();
  if (!PyObject_HasAttrString (core_obj, (char *)"getOutCaps")) {
    ml_loge ("Cannot find 'getOutCaps'");
    ml_loge ("defualt caps is `application/octet-stream`");
    caps = gst_caps_from_string ("application/octet-stream");
    goto done;
  }

  result = PyObject_CallMethod (core_obj, (char *)"getOutCaps", NULL);
  if (result) {
    gchar * caps_str = PyBytes_AsString (result);
    caps = gst_caps_from_string (caps_str);
    Py_SAFEDECREF (result);
  } else {
    caps = gst_caps_from_string ("application/octet-stream");
  }

done:
  Py_UNLOCK ();
  return caps;
}

/**
 * @brief	initialize the object with python script
 */
int
PYDecoderCore::init ()
{
  /** Find nnstreamer_api module */
  PyObject *api_module = PyImport_ImportModule ("nnstreamer_python");
  if (api_module == NULL) {
    return -EINVAL;
  }

  shape_cls = PyObject_GetAttrString (api_module, "TensorShape");
  Py_SAFEDECREF (api_module);

  if (shape_cls == NULL)
    return -EINVAL;

  return loadScript (&core_obj, module_name.c_str(), "CustomDecoder");
}

/**
 * @brief	get the script path
 * @return the script path.
 */
const char *
PYDecoderCore::getScriptPath ()
{
  return script_path.c_str ();
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
decoder_py_init (void **pdata)
{
  UNUSED (pdata);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
decoder_py_exit (void **pdata)
{
  PYDecoderCore *core = static_cast<PYDecoderCore *> (*pdata);

  g_return_if_fail (core != NULL);
  delete core;

  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
decoder_py_setOption (void **pdata, int opNum, const char *param)
{
  gchar *path = (gchar *) param;

  /* opNum 1 = python script path */
  if (opNum == 0) {
    PYDecoderCore *core;

    if (!Py_IsInitialized ())
      throw std::runtime_error ("Python is not initialize.");

    /** Load python script file */
    core = static_cast<PYDecoderCore *> (*pdata);

    if (core != NULL) {
      if (g_strcmp0 (path, core->getScriptPath ()) == 0)
        return TRUE; /* skipped */

      decoder_py_exit (pdata);
    }

    /* init null */
    *pdata = NULL;

    try {
      core = new PYDecoderCore (path);
    } catch (std::bad_alloc & exception) {
      ml_loge ("Failed to allocate memory for decoder subplugin: python3\n");
      ml_loge ("%s", exception.what());
      return FALSE;
    }

    if (core->init () != 0) {
      delete core;
      ml_loge ("failed to initailize the object: Python3\n");
      return FALSE;
    }
    *pdata = core;

    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
decoder_py_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  GstCaps *caps;
  PYDecoderCore *core = static_cast<PYDecoderCore *> (*pdata);

  caps = core->getOutCaps (config);
  setFramerateFromConfig (caps, config);
  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
decoder_py_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  PYDecoderCore *core = static_cast<PYDecoderCore *> (*pdata);
  g_return_val_if_fail (config, GST_FLOW_ERROR);
  g_return_val_if_fail (input, GST_FLOW_ERROR);
  g_return_val_if_fail (outbuf, GST_FLOW_ERROR);
  return core->decode (config, input, outbuf);
}

static gchar decoder_subplugin_python3[] = "python3";

/** @brief python3fer tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef Python = { .modename = decoder_subplugin_python3,
  .init = decoder_py_init,
  .exit = decoder_py_exit,
  .setOption = decoder_py_setOption,
  .getOutCaps = decoder_py_getOutCaps,
  .decode = decoder_py_decode,
  .getTransformSize = NULL };

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** @brief Initialize this object for tensordec-plugin */
void
init_decoder_py (void)
{
  /** Python should be initialized and finalized only once */
  if (!Py_IsInitialized ())
    Py_Initialize ();

  nnstreamer_decoder_probe (&Python);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_decoder_py (void)
{
  nnstreamer_decoder_exit (Python.modename);

  /** Python should be initialized and finalized only once */
  if (Py_IsInitialized ())
    Py_Finalize ();
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
