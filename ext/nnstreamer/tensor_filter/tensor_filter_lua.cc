/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, LUA Script Module
 * Copyright (C) 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file        tensor_filter_lua.cc
 * @date        25 Feb 2021
 * @brief       LUA script loading module for tensor_filter
 * @see         https://nnstreamer.github.io
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 *              Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug         NYI
 *
 * @detail
 *      Users are supposed to supply LUA scripts processing
 *  given input tensors.
 *
 *   With file mode, a file path to the script file is
 * specified. To update its contents, the file path property
 * should be updated.
 *
 *   With script mode, the script is specified as a property
 * value, which can be updated in run-time.
 *
 *   The input and output dimensions may be * constant,
 * designated by global LUA variables, "inputTensorsInfo"
 * and "outputTensorsInfo".
 *
 *   Both input/outputTensorsInfo are required to have:
 * {
 *    // Array starts from 1, ends at num
 *    num, // Number of tensors (1 .. 16)
 *    type = string[num], // TYPE (INT32, INT64, UINT8, FLOAT32, ...)
 *    dim = int[num][4], // Dimension {1, 2, 3, 4} == 1 : 2 : 3 : 4
 * }
 *   An Example:
 * inputTensorsInfo = {
 *  num = 1,
 *  dim = {{3, 640, 480, 1}, },
 *  type = {'uint8', }
 * }
 * outputTensorsInfo = {
 *  num = 1,
 *  dim = {{3, 640, 480, 1}, },
 *  type = {'uint8', }
 * }
 *
 *   Then, users need to provide a global "nnstreamer_invoke()" function,
 * where the function can get value of input tensors and set value of
 * output tensors. We offer two APIs for access those tensors:
 * "input_tensor(tensor_idx)" and "output_tensor(tensor_idx)"
 *
 *   An Example:
 * function nnstreamer_invoke()
 *   oC = outputTensorsInfo['dim'][1][1]
 *   oW = outputTensorsInfo['dim'][1][2]
 *   oH = outputTensorsInfo['dim'][1][3]
 *   oN = outputTensorsInfo['dim'][1][4]
 *
 *   input = input_tensor(1)   --[[ get first input tensor --]]
 *   output = output_tensor(1) --[[ get first output tensor --]]
 *   for i=1,oC*oW*oH*oN do
 *     output[i] = input[i] --[[ copy input to output --]]
 *   end
 * end
 *
 *   In "script mode", not "file mode", the script should NOT have
 * double quote ("), and double dashes ( -- COMMENT ) for comment.
 * Use single quote and --[[ COMMENT --]] format instead.
 *
 */

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <glib.h>
#include <string>
#include <memory>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>


namespace nnstreamer
{
namespace tensorfilter_lua
{
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void _init_filter_lua (void) __attribute__((constructor));
void _fini_filter_lua (void) __attribute__((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief Data for lua tensor
 */
typedef struct lua_tensor {
  tensor_type type;
  void *data;
  size_t size;
} lua_tensor;

/** @brief For getting value in Lua */
static int
tensor_index (lua_State *L)
{
  lua_tensor *lt = *((lua_tensor **) luaL_checkudata(L, 1, "lua_tensor"));
  int tidx = luaL_checkint (L, 2) - 1;

  uint element_size = gst_tensor_get_element_size (lt->type);
  if (tidx < 0 || (size_t) tidx * element_size >= lt->size)
    throw std::runtime_error ("Invalid index for tensor");

  double value = 0.0;
  switch (lt->type) {
    case _NNS_INT32:
      value = (double) ((int32_t *) lt->data)[tidx];
      break;
    case _NNS_UINT32:
      value = (double) ((uint32_t *) lt->data)[tidx];
      break;
    case _NNS_INT16:
      value = (double) ((int16_t *) lt->data)[tidx];
      break;
    case _NNS_UINT16:
      value = (double) ((uint16_t *) lt->data)[tidx];
      break;
    case _NNS_INT8:
      value = (double) ((int8_t *) lt->data)[tidx];
      break;
    case _NNS_UINT8:
      value = (double) ((uint8_t *) lt->data)[tidx];
      break;
    case _NNS_FLOAT64:
      value = (double) ((double *) lt->data)[tidx];
      break;
    case _NNS_FLOAT32:
      value = (double) ((float *) lt->data)[tidx];
      break;
    case _NNS_FLOAT16:
#ifdef FLOAT16_SUPPORT
      value = (double) ((float16 *) lt->data)[tidx];
#else
      nns_loge
          ("NNStreamer requires -DFLOAT16_SUPPORT as a build option to enable float16 type. This binary does not have float16 feature enabled; thus, float16 type is not supported in this instance.\n");
      throw std::runtime_error ("Float16 not supported. Recompile with -DFLOAT16_SUPPORT.");
#endif
      break;
    case _NNS_INT64:
      value = (double) ((int64_t *) lt->data)[tidx];
      break;
    case _NNS_UINT64:
      value = (double) ((uint64_t *) lt->data)[tidx];
      break;
    default:
      throw std::runtime_error ("Error occurred during get tensor value");
      break;
  }

  lua_pushnumber (L, value);

  return 1;
}

/** @brief For assigning new value in Lua */
static int
tensor_newindex (lua_State* L)
{
  lua_tensor *lt = *((lua_tensor **) luaL_checkudata (L, 1, "lua_tensor"));
  int tidx = luaL_checkint(L, 2) - 1;
  double value = luaL_checknumber (L, 3);

  uint element_size = gst_tensor_get_element_size (lt->type);
  if (tidx < 0 || (size_t) tidx * element_size >= lt->size)
    throw std::runtime_error ("Invalid index for tensor");

  switch (lt->type) {
    case _NNS_INT32:
      ((int32_t *) lt->data)[tidx] = (int32_t) value;
      break;
    case _NNS_UINT32:
    {
      int32_t temp = (int32_t) value;
      ((uint32_t *) lt->data)[tidx] = (uint32_t) temp;
      break;
    }
    case _NNS_INT16:
      ((int16_t *) lt->data)[tidx] = (int16_t) value;
      break;
    case _NNS_UINT16:
    {
      int16_t temp = (int16_t) value;
      ((uint16_t *) lt->data)[tidx] = (uint16_t) temp;
      break;
    }
    case _NNS_INT8:
      ((int8_t *) lt->data)[tidx] = (int8_t) value;
      break;
    case _NNS_UINT8:
    {
      int8_t temp = (int8_t) value;
      ((uint8_t *) lt->data)[tidx] = (uint8_t) temp;
      break;
    }
    case _NNS_FLOAT64:
      ((double *) lt->data)[tidx] = (uint8_t) value;
      break;
    case _NNS_FLOAT32:
      ((float *) lt->data)[tidx] = (float) value;
      break;
    case _NNS_FLOAT16:
#ifdef FLOAT16_SUPPORT
      ((float16 *) lt->data)[tidx] = (float16) value;
#else
      nns_loge
          ("NNStreamer requires -DFLOAT16_SUPPORT as a build option to enable float16 type. This binary does not have float16 feature enabled; thus, float16 type is not supported in this instance.\n");
      throw std::runtime_error ("Float16 not supported. Recompile with -DFLOAT16_SUPPORT.");
#endif
      break;
    case _NNS_INT64:
      ((int64_t *) lt->data)[tidx] = (int64_t) value;
      break;
    case _NNS_UINT64:
    {
      int64_t temp = (int64_t) value;
      ((uint64_t *) lt->data)[tidx] = (uint64_t) temp;
      break;
    }
    default:
      throw std::runtime_error ("Error occurred during set tensor value");
      break;
  }

  return 0;
}

/** @brief Expose C array to Lua */
static int
expose_tensor (lua_State* L, lua_tensor *tensor)
{
  lua_tensor **ptensor = (lua_tensor **) lua_newuserdata (L, sizeof (lua_tensor *));
  *ptensor = tensor;
  luaL_getmetatable (L, "lua_tensor");
  lua_setmetatable (L, -2);

  return 1;
}

/** @brief Get input tensor in Lua */
static int
getInputTensor (lua_State* L)
{
  int tidx = lua_tointeger (L, 1);
  if (tidx <= 0 || tidx > NNS_TENSOR_SIZE_LIMIT) {
    throw std::runtime_error ("Invalid idx for `input_tensor(idx)`");
  }

  lua_pushstring (L, "input_lua_tensors");
  lua_gettable (L, LUA_REGISTRYINDEX);
  lua_tensor *lt = (lua_tensor *) lua_topointer (L, -1);

  return expose_tensor (L, &(lt[tidx - 1]));
}

/** @brief Get output tensor in Lua */
static int
getOutputTensor (lua_State *L)
{
  int tidx = lua_tointeger (L, 1);
  if (tidx <= 0 || tidx > NNS_TENSOR_SIZE_LIMIT) {
    throw std::runtime_error ("Invalid idx for `output_tensor(idx)`");
  }

  lua_pushstring (L, "output_lua_tensors");
  lua_gettable (L, LUA_REGISTRYINDEX);
  lua_tensor *lt = (lua_tensor *) lua_topointer (L, -1);

  return expose_tensor (L, &(lt[tidx - 1]));
}

/** @brief Register metatable for tensor in Lua */
static void
create_tensor_type (lua_State *L)
{
  static const struct luaL_reg tensor[] = {
    {"__index", tensor_index},
    {"__newindex", tensor_newindex},
    {NULL, NULL}
  };

  luaL_newmetatable (L, "lua_tensor");
  luaL_openlib (L, NULL, tensor, 0);
  lua_register (L, "input_tensor", getInputTensor);
  lua_register (L, "output_tensor", getOutputTensor);
}

/** @brief lua subplugin class */
class lua_subplugin final : public tensor_filter_subplugin
{
  private:
  static const char *name;
  static lua_subplugin *registeredRepresentation;
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  lua_State *L;
  static const accl_hw hw_list[];
  static const int num_hw = 1;

  lua_tensor input_lua_tensors[NNS_TENSOR_SIZE_LIMIT];
  lua_tensor output_lua_tensors[NNS_TENSOR_SIZE_LIMIT];

  void setTensorsInfo (GstTensorsInfo &tensors_info);

  public:
  static void init_filter_lua ();
  static void fini_filter_lua ();

  lua_subplugin ();
  ~lua_subplugin ();

  /** implementing virtual methods of parent */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *lua_subplugin::name = "lua";
const accl_hw lua_subplugin::hw_list[] = { ACCL_CPU };

/** @brief Class constructor */
lua_subplugin::lua_subplugin ()
    : tensor_filter_subplugin (), L (NULL), input_lua_tensors {}, output_lua_tensors {}
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
}

/** @brief Class destructor */
lua_subplugin::~lua_subplugin ()
{
  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  if (L != NULL)
    lua_close (L);
}

/** @brief tensor-filter subplugin mandatory method */
tensor_filter_subplugin &
lua_subplugin::getEmptyInstance  ()
{
  return *(new lua_subplugin ());
}

/** @brief Parse TensorsInfo from the given Lua script */
void
lua_subplugin::setTensorsInfo (GstTensorsInfo &tensors_info)
{

  if (lua_istable (L, -1)) {
    lua_pushstring (L, "num");
    lua_gettable (L, -2);
    if (lua_isnumber (L, -1)) {
      int num_tensors = lua_tointeger (L, -1);
      if (num_tensors <= 0 || num_tensors > NNS_TENSOR_SIZE_LIMIT)
        throw std::invalid_argument (
            "The number of tensors required by the given model exceeds the nnstreamer tensor limit (" NNS_TENSOR_SIZE_LIMIT_STR " by default).");
      tensors_info.num_tensors = (uint) num_tensors;
    } else {
      throw std::invalid_argument ("Failed to parse `num`. Please check the script");
    }
    lua_pop (L, 1);

    lua_pushstring (L, "type");
    lua_gettable (L, -2);
    if (lua_istable (L, -1)) {
      for (uint j = 1; j <= tensors_info.num_tensors; ++j) {
        lua_pushinteger (L, j);
        lua_gettable (L, -2);
        tensors_info.info[j - 1].type = gst_tensor_get_type (lua_tostring (L, -1));
        if (tensors_info.info[j - 1].type == _NNS_END)
          throw std::invalid_argument ("Failed to parse `type`. Possible types are " GST_TENSOR_TYPE_ALL);
        lua_pop (L, 1);
      }
    } else {
      throw std::invalid_argument ("Failed to parse `type`. Please check the script");
    }
    lua_pop (L, 1);

    lua_pushstring (L, "dim");
    lua_gettable (L, -2);
    if (lua_istable (L, -1)) {
      for (uint j = 1; j <= tensors_info.num_tensors; ++j) {
        lua_pushinteger (L, j);
        lua_gettable (L, -2);
        if (lua_istable (L, -1)) {
          uint len = lua_objlen (L, -1);
          for (uint i = 1; i <= len; ++i) {
            lua_pushinteger (L, i);
            lua_gettable (L, -2);
            if (lua_isnumber (L, -1)) {
              tensors_info.info[j - 1].dimension[i - 1] = lua_tointeger (L, -1);
            } else {
              throw std::invalid_argument ("Failed to parse `dim`. Please check the script");
            }
            lua_pop (L, 1);
          }
          for (uint i = len + 1; i <= NNS_TENSOR_RANK_LIMIT; i++) {
            tensors_info.info[j - 1].dimension[i - 1] = 1;
          }
        } else {
          throw std::invalid_argument ("Failed to parse `dim`. Please check the script");
        }
        lua_pop (L, 1);
      }
    } else {
      throw std::invalid_argument ("Failed to parse `dim`. Please check the script");
    }
    lua_pop (L, 1);
  } else {
    throw std::invalid_argument ("Failed to parse global variable `[input/output]TensorsInfo`. Please check the script");
  }
}

/** @brief tensor-filter subplugin mandatory method */
void
lua_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (L != NULL) {
    lua_close (L);
  }

  L = lua_open ();
  luaL_openlibs (L);

  lua_pushstring (L, "input_lua_tensors");
  lua_pushlightuserdata (L, (void *) input_lua_tensors);
  lua_settable (L, LUA_REGISTRYINDEX);

  lua_pushstring (L, "output_lua_tensors");
  lua_pushlightuserdata (L, (void *) output_lua_tensors);
  lua_settable (L, LUA_REGISTRYINDEX);

  create_tensor_type (L);

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_EXISTS)) {
    nns_logi ("Given model file does not exist. Do script mode.");
    gchar *script = g_strjoinv (",", (gchar **) prop->model_files);
    std::unique_ptr<gchar, decltype (&g_free)> script_ptr (script, g_free);

    if (luaL_dostring (L, script_ptr.get ()) != 0) {
      throw std::invalid_argument (std::string ("Failed to run given Lua script. Error message: ") +
          lua_tostring (L, -1));
    }
  } else {
    /** Do File mode */
    if (luaL_dofile (L, prop->model_files[0]) != 0) {
      throw std::invalid_argument (std::string ("Failed to run given Lua script file. Error message: ") +
          lua_tostring (L, -1));
    }
  }

  lua_getglobal (L, "inputTensorsInfo");
  setTensorsInfo (inputInfo);
  lua_getglobal (L, "outputTensorsInfo");
  setTensorsInfo (outputInfo);

  lua_getglobal (L, "nnstreamer_invoke");
  if (!lua_isfunction (L, -1)) {
    throw std::invalid_argument ("Error while loading function `nnstreamer_invoke` in lua script");
  }

  lua_settop (L, 0);
}

/** @brief tensor-filter subplugin mandatory method */
void
lua_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  for (uint i = 0; i < inputInfo.num_tensors; ++i) {
    input_lua_tensors[i].type = inputInfo.info[i].type;
    input_lua_tensors[i].data = input[i].data;
    input_lua_tensors[i].size = input[i].size;
  }

  for (uint i = 0; i < outputInfo.num_tensors; ++i) {
    output_lua_tensors[i].type = outputInfo.info[i].type;
    output_lua_tensors[i].data = output[i].data;
    output_lua_tensors[i].size = output[i].size;
  }

  lua_getglobal (L, "nnstreamer_invoke");
  if (lua_isfunction(L, -1)) {
      if (lua_pcall (L, 0, 0, 0) != 0) {
        throw std::runtime_error ("error while pcall nnstreamer_invoke");
      }
  } else {
    throw std::runtime_error ("Error while loading function `nnstreamer_invoke` in lua script");
  }
}

/** @brief tensor-filter subplugin mandatory method */
void
lua_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 0; /** We may run scripts expressed as property */
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

/** @brief tensor-filter subplugin mandatory method */
int
lua_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/** @brief tensor-filter subplugin mandatory method */
int
lua_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

lua_subplugin *lua_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Initialize the object for runtime register
 */
void
lua_subplugin::init_filter_lua (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<lua_subplugin> ();
}

/**
 * @brief Destruct the subplugin
 */
void
lua_subplugin::fini_filter_lua (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief initializer
 */
void
_init_filter_lua ()
{
  lua_subplugin::init_filter_lua ();
}

/**
 * @brief finalizer
 */
void
_fini_filter_lua ()
{
  lua_subplugin::fini_filter_lua ();
}

} /* namespace nnstreamer::tensorfilter_lua */
} /* namespace nnstreamer */
