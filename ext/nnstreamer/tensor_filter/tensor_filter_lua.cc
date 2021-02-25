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
 * @bug         NYI
 *
 * @detail
 *      Users are supposed to supply LUA scripts processing
 *  given input tensors.
 *
 *      With file mode, a file path to the script file is
 * specified. To update its contents, the file path property
 * should be updated. (or send "reload" event. method: TBD)
 *
 *      With script mode, the script is specified as a property
 * value, which can be updated in run-time.
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


namespace nnstreamer
{
namespace tensorfilter_lua
{

void _init_filter_lua (void) __attribute__ ((constructor));
void _fini_filter_lua (void) __attribute__ ((destructor));

/** @brief lua subplugin class */
class lua_subplugin final : public tensor_filter_subplugin
{
  private:
  static const char *name;
  const char *script;
  const char *loadScript (const GstTensorFilterProperties *prop);
  lua_State *L;
  static const accl_hw hw_list[];
  static const int num_hw = 1;

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
    : tensor_filter_subplugin (), script (NULL), L (NULL)
{
}

/** @brief Class destructor */
lua_subplugin::~lua_subplugin ()
{
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

  L = luaL_newstate();
  luaL_openlibs (L);

  script = loadScript (prop);
  int load_stat = luaL_loadbuffer (L, script, strlen (script), script);
  if (load_stat != 0) {
    /** @todo Error handling with load_stat */
    return;
  }

  lua_pcall(L, 0, 0, 0); /** execute "script" to load the "invoke" func. */

  lua_getglobal (L, "nnstreamer_invoke");
  if (lua_isfunction (L, -1)) {
    /** "OK" */
    /** @todo handle and return */
  } else {
    /** @todo Error handling. */
  }

}

/** @brief tensor-filter subplugin mandatory method */
void
lua_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  /** @todo STEP 1. Prepare input buffer for LUA */

  /** @todo STEP 2. Prepare output buffer for LUA */

  /** STEP 3. Call LUA
    * @todo NYI input/output handling.
    * @todo NYI LUA APIs for tensor-filter
    *
    * Suggestion: set "standard global invoke function name to be called",
    * , which is "nnstreamer_invoke". The next lua_pcall will call that
    * function.
    */
  lua_pcall (L, 0, 0, 0); /** @todo NYI. replace 0 with correct variables */

  /** @todo STEP 4. Anything else? */
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
  /** @todo Need some connection with lua script? or let property handle? */
  return -ENOENT;
}

/** @brief tensor-filter subplugin mandatory method */
int
lua_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  /** @todo Handle "reload" */
  return -ENOENT;
}

/** @brief Load lua script based on prop info */
const char *
lua_subplugin::loadScript (const GstTensorFilterProperties *prop)
{
  /** @todo NYI */
  return NULL;
}


} /* namespace nnstreamer::tensorfilter_lua */
} /* namespace nnstreamer */
