/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_converter subplugin, "python3"
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	tensor_converter_python3.c
 * @date	May 03 2021
 * @brief	NNStreamer tensor-converter subplugin, "python3",
*         which converts to tensors using python.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		python converter with Python3.9.10 is stucked during Py_Finalize().
 */

#include <nnstreamer_plugin_api_converter.h>
#include <nnstreamer_util.h>
#include "nnstreamer_python3_helper.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_converter_py (void) __attribute__((constructor));
void fini_converter_py (void) __attribute__((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief	Python embedding core structure
 */
class PYConverterCore
{
  public:
  /**
   * member functions.
   */
  PYConverterCore (const char *_script_path);
  ~PYConverterCore ();

  int init ();
  const char *getScriptPath ();
  GstBuffer *convert (GstBuffer*in_buf, GstTensorsConfig *config);

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
 * @brief	PYConverterCore creator
 * @param	_script_path	: the logical path to '{script_name}.py' file
 * @note	the script of _script_path will be loaded simultaneously
 * @return	Nothing
 */
PYConverterCore::PYConverterCore (const char *_script_path)
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
  const size_t last_idx = module_name.find_last_of ("/");

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
 * @brief	PYConverterCore Destructor
 * @return	Nothing
 */
PYConverterCore::~PYConverterCore ()
{
  Py_SAFEDECREF (core_obj);
  Py_SAFEDECREF (shape_cls);
  PyErr_Clear ();

  dlclose (handle);
  g_mutex_clear (&py_mutex);
}

/**
 * @brief	convert any media stream to tensor
 */
GstBuffer *
PYConverterCore::convert (GstBuffer *in_buf, GstTensorsConfig *config)
{
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT], *out_mem;
  GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
  GstBuffer *out_buf = NULL;
  PyObject *tensors_info, *output, *pyValue, *param;
  gint rate_n, rate_d;
  guint i, num;
  gsize mem_size;
  gpointer mem_data;

  if (nullptr == in_buf)
    throw std::invalid_argument ("Null pointers are given to PYConverterCore::convert().\n");
  num = gst_buffer_n_memory (in_buf);
  tensors_info = output = pyValue = param = nullptr;

  Py_LOCK ();
  param = PyList_New (num);

  for (i = 0; i < num; i++) {
    in_mem[i] = gst_buffer_peek_memory (in_buf, i);

    if (!gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
      Py_ERRMSG ("Cannot map input memory / tensor_converter::custom-script");
      num = i;
      goto done;
    }

    npy_intp input_dims[] = { (npy_intp) (in_info[i].size) };
    PyObject *input_array = PyArray_SimpleNewFromData (
        1, input_dims, NPY_UINT8, in_info[i].data);

    PyList_SetItem (param, i, input_array);
  }

  if (!PyObject_HasAttrString (core_obj, (char *)"convert")) {
    Py_ERRMSG ("Cannot find 'convert'");
    goto done;
  }

  pyValue = PyObject_CallMethod (core_obj, "convert", "(O)", param);

  if (!PyArg_ParseTuple (pyValue, "OOii", &tensors_info, &output,
      &rate_n, &rate_d)) {
    Py_ERRMSG ("Failed to parse converting result");
    goto done;
  }

  if (parseTensorsInfo (tensors_info, &config->info) != 0) {
    Py_ERRMSG ("Failed to parse tensors info");
    goto done;
  }
  config->rate_n = rate_n;
  config->rate_d = rate_d;

  if (output) {
    unsigned int num_tensors = PyList_Size (output);

    out_buf = gst_buffer_new ();
    for (unsigned int i = 0; i < num_tensors; i++) {
      PyArrayObject *output_array
          = (PyArrayObject *)PyList_GetItem (output, (Py_ssize_t)i);

      mem_size = PyArray_SIZE (output_array);
      mem_data = _g_memdup ((guint8 *) PyArray_DATA (output_array), mem_size);

      out_mem = gst_memory_new_wrapped ((GstMemoryFlags) 0, mem_data, mem_size,
          0, mem_size, mem_data, g_free);
      gst_buffer_append_memory (out_buf, out_mem);
    }
  } else {
    Py_ERRMSG ("Fail to get output from 'convert'");
  }

done:
  for (i = 0; i < num; i++)
    gst_memory_unmap (in_mem[i], &in_info[i]);

  Py_SAFEDECREF (param);
  Py_SAFEDECREF (pyValue);

  Py_UNLOCK ();
  return out_buf;
}

/**
 * @brief	initialize the object with python script
 * @return 0 if OK. non-zero if error.
 */
int
PYConverterCore::init ()
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

  return loadScript (&core_obj, module_name.c_str(), "CustomConverter");
}

/**
 * @brief	get the script path
 * @return the script path.
 */
const char *
PYConverterCore::getScriptPath ()
{
  return script_path.c_str ();
}

/**
 * @brief Free privateData and move on.
 */
static void
py_close (void **private_data)
{
  PYConverterCore *core = static_cast<PYConverterCore *> (*private_data);

  g_return_if_fail (core != NULL);
  delete core;

  *private_data = NULL;
}

/**
 * @brief The open callback for GstTensorConverterFramework. Called before anything else
 * @param path: python script path
 * @param private_data: python plugin's private data
 */
static int
py_open (const gchar *path, void **priv_data)
{
  PYConverterCore *core;

  if (!Py_IsInitialized ())
    throw std::runtime_error ("Python is not initialize.");

  /** Load python script file */
  core = static_cast<PYConverterCore *> (*priv_data);

  if (core != NULL) {
    if (g_strcmp0 (path, core->getScriptPath ()) == 0)
      return 1; /* skipped */

    py_close (priv_data);
  }

  /* init null */
  *priv_data = NULL;

  core = new PYConverterCore (path);
  if (core == NULL) {
    Py_ERRMSG ("Failed to allocate memory for converter subplugin: Python\n");
    return -1;
  }

  if (core->init () != 0) {
    delete core;
    Py_ERRMSG ("failed to initailize the object: Python\n");
    return -2;
  }
  *priv_data = core;

  return 0;
}

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstCaps *
python_query_caps (const GstTensorsConfig *config)
{
  UNUSED (config);
  return gst_caps_from_string ("application/octet-stream");
}

/**
 * @brief tensor converter plugin's NNStreamerExternalConverter callback
 */
static gboolean
python_get_out_config (const GstCaps *in_cap, GstTensorsConfig *config)
{
  GstStructure *structure;
  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensors_config_init (config);
  g_return_val_if_fail (in_cap != NULL, FALSE);

  structure = gst_caps_get_structure (in_cap, 0);
  g_return_val_if_fail (structure != NULL, FALSE);

  /* All tensor info should be updated later in chain function. */
  config->info.info[0].type = _NNS_UINT8;
  config->info.num_tensors = 1;
  if (gst_tensor_parse_dimension ("1:1:1:1", config->info.info[0].dimension) == 0) {
    Py_ERRMSG ("Failed to set initial dimension for subplugin");
    return FALSE;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n, &config->rate_d);
  } else {
    /* cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }
  return TRUE;
}

/**
 * @brief tensor converter plugin's NNStreamerExternalConverter callback
 */
static GstBuffer *
python_convert (GstBuffer *in_buf, GstTensorsConfig *config, void *priv_data)
{
  PYConverterCore *core = static_cast<PYConverterCore *> (priv_data);
  g_return_val_if_fail (in_buf, NULL);
  g_return_val_if_fail (config, NULL);
  return core->convert (in_buf, config);
}

static const gchar converter_subplugin_python[] = "python3";

/** @brief flatbuffer tensor converter sub-plugin NNStreamerExternalConverter instance */
static NNStreamerExternalConverter Python = {
  .name = converter_subplugin_python,
  .convert = python_convert,
  .get_out_config = python_get_out_config,
  .query_caps = python_query_caps,
  .open = py_open,
  .close = py_close,
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/** @brief Initialize this object for tensor converter sub-plugin */
void
init_converter_py (void)
{
  /** Python should be initialized and finalized only once */
  if (!Py_IsInitialized()) {
    Py_Initialize ();
  }
  registerExternalConverter (&Python);
}

/** @brief Destruct this object for tensor converter sub-plugin */
void
fini_converter_py (void)
{
  unregisterExternalConverter (Python.name);
/**
 * @todo Remove below lines after this issue is addressed.
 * Tizen issues: After python version has been upgraded from 3.9.1 to 3.9.10, python converter is stopped at Py_Finalize.
 * Since Py_Initialize is not called twice from this object, Py_Finalize is temporarily removed.
 */
#if 0
  /** Python should be initialized and finalized only once */
  if (Py_IsInitialized())
    Py_Finalize ();
#endif
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
