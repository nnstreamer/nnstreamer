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
 * should be updated. (or send "reload" event. method: TBD)
 *
 *   With script mode, the script is specified as a property
 * value, which can be updated in run-time.
 *
 *   The input and output dimensions may be * constant,
 * designated by global LUA variables, "inputConf"
 * and "outputConf".
 *
 *   Or the dimensions may be provided by the following
 * global functions, which have LOWER priority than
 * the constants.
 * - getInputOutputConf ()
 * - getOutputConfFromInputConf ()
 *
 *   Then, users need to provide a global "nnstreamer_invoke" function,
 * "invokeNN", where the function type is:
 *   @todo scale for num_tensors > 1
 *         Support various tensor type
 *
 *   Both input/outputConf are required to have:
 * {
 *    // Array starts from 1, ends at num
 *    num, // Number of tensors (1 .. 16)
 *    types = string[num], // TYPES (INT32, INT64, ...)
 *    dims = int[num][4], // Dimension { 1, 2, 3} == 1 : 2 : 3
 *    names = string[num], // The optional tensor names.
 * }
 */

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <glib.h>
#include <string>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
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

/** @brief For getting value in Lua */
static int
tensor_index (lua_State *L)
{
  uint8_t** parray = (uint8_t **) luaL_checkudata (L, 1, "tensor");
  int index = luaL_checkint (L, 2);
  lua_pushnumber (L, (*parray)[index - 1]);

  return 1;
}

/** @brief For assigning new value in Lua */
static int
tensor_newindex (lua_State* L)
{
  uint8_t** parray = (uint8_t **) luaL_checkudata(L, 1, "tensor");
  int index = luaL_checkint(L, 2);
  int value = luaL_checkint(L, 3);
  (*parray)[index - 1] = value;

  return 0;
}

/** @brief Get tensor value in Lua */
static int
expose_tensor (lua_State*L, uint8_t *array)
{
  uint8_t** parray = (uint8_t **) lua_newuserdata (L, sizeof (uint8_t **));
  *parray = array;
  luaL_getmetatable (L, "tensor");
  lua_setmetatable (L, -2);

  return 1;
}

/** @brief Get input tensor in Lua */
static int
getInputTensor (lua_State* L)
{
  lua_pushstring (L, "input_for_lua");
  lua_gettable (L, LUA_REGISTRYINDEX);
  const GstTensorMemory **res = (const GstTensorMemory **) lua_topointer (L, -1);
  lua_remove (L, 1);

  return expose_tensor (L, (uint8_t *) (*res)[0].data);
}

/** @brief Get output tensor in Lua */
static int
getOutputTensor (lua_State *L)
{
  lua_pushstring (L, "output_for_lua");
  lua_gettable (L, LUA_REGISTRYINDEX);
  GstTensorMemory **res = (GstTensorMemory **) lua_topointer (L, -1);
  lua_remove (L, 1);

  return expose_tensor (L, (uint8_t *) (*res)[0].data);
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

  luaL_newmetatable (L, "tensor");
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

  const GstTensorMemory *input_for_lua;
  GstTensorMemory *output_for_lua;

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
    : tensor_filter_subplugin (), L (NULL)
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

/** @brief tensor-filter subplugin mandatory method */
void
lua_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (L != NULL) {
    lua_close (L);
  }

  /** @todo scale for num_tensors > 1 */
  inputInfo.num_tensors = 1;
  outputInfo.num_tensors = 1;

  L = lua_open ();
  luaL_openlibs (L);

  lua_pushstring (L, "input_for_lua");
  lua_pushlightuserdata (L, (void *) &input_for_lua);
  lua_settable (L, LUA_REGISTRYINDEX);

  lua_pushstring (L, "output_for_lua");
  lua_pushlightuserdata (L, (void *) &output_for_lua);
  lua_settable (L, LUA_REGISTRYINDEX);

  create_tensor_type (L);

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_EXISTS)) {
    nns_logi ("Given model file does not exist. Do script mode.");
    std::string script = g_strjoinv (",", (gchar **) prop->model_files);
    if (luaL_dostring (L, script.c_str ()) != 0) {
      nns_loge ("Error occured while loading given lua script: %s\nError message: %s",
          script.c_str (), lua_tostring (L, -1));

      return;
    }
  } else {
    /** Do File mode */
    if (luaL_dofile (L, prop->model_files[0]) != 0) {
      nns_loge ("Error occured while loading given lua script.\nError message: %s",
          lua_tostring (L, -1));

      return ;
    }
  }

  /** Parsing inputTensorInfo */
  lua_getglobal (L, "inputTensorInfo");
  if (lua_istable (L, -1)) {
    lua_pushstring (L, "type");
    lua_gettable (L, -2);

    /** @todo Support various tensor type */
    std::string lua_input_type = lua_tostring (L, -1);
    inputInfo.info[0].type = _NNS_UINT8;
    lua_pop (L, 1);

    lua_pushstring (L, "dim");
    lua_gettable (L, -2);
    if (lua_istable (L, -1)) {
      for (int i = 1; i <= 4; ++i) {
        lua_pushinteger (L, i);
        lua_gettable (L, -2);
        inputInfo.info[0].dimension[i - 1] = lua_tointeger (L, -1);
        lua_pop (L, 1);
      }

      lua_pop (L, 1);
    }
    lua_pop (L, 1);
  } else {
    nns_loge ("Failed to get inputTensorInfo from lua. Please check the script");
  }

  /** Parsing outputTensorInfo */
  lua_getglobal (L, "outputTensorInfo");
  if (lua_istable (L, -1)) {
    lua_pushstring (L, "type");
    lua_gettable (L, -2);

    /** @todo Support various tensor type */
    std::string lua_output_type = lua_tostring (L, -1);
    outputInfo.info[0].type = _NNS_UINT8;
    lua_pop (L, 1);

    lua_pushstring (L, "dim");
    lua_gettable (L, -2);
    if (lua_istable (L, -1)) {
      for (int i = 1; i <= 4; ++i) {
        lua_pushinteger (L, i);
        lua_gettable (L, -2);
        outputInfo.info[0].dimension[i - 1] = lua_tointeger (L, -1);
        lua_pop (L, 1);
      }
      lua_pop (L, 1);
    }
    lua_pop (L, 1);
  } else {
    nns_loge ("Failed to get OutputTensorInfo from lua. Please check the script");
  }

  lua_getglobal (L, "nnstreamer_invoke");
  if (!lua_isfunction (L, -1)) {
    nns_loge ("Error while loading function `nnstreamer_invoke` in lua script");
  }

  lua_settop (L, 0);
}

/** @brief tensor-filter subplugin mandatory method */
void
lua_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  input_for_lua = input;
  output_for_lua = output;

  lua_getglobal (L, "nnstreamer_invoke");
  if (lua_isfunction(L, -1)) {
      if (lua_pcall (L, 0, 0, 0) != 0) {
        nns_logw ("error while pcall nnstreamer_invoke");
      }
  } else {
    nns_loge ("Error while loading function `nnstreamer_invoke` in lua script");
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
  /** @todo Handle "reload" */
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
