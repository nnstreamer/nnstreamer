/**
 * GStreamer Tensor_Filter, Sub-plugin for Query
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    tensor_filter_query.cc
 * @date    07 Nov 2020
 * @brief   Tensor_filter subplugin for Query.
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Query) for tensor_filter. Note that 'Query'
 * does not mean a specific NN-framework but provides the remote query capability.
 *
 * This sends queries to server pipelines through the gRPC interface, which accept
 * the queries and may perform interesting work on behalf of the original pipeline.
 * This provides efficient compute-offloading and data communication in Edge-AI env.
 */

#include <iostream>
#include <string>

#include <glib.h>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_grpc.h>

#include <tensor_common.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

namespace nnstreamer
{
namespace tensor_filter_query
{

extern "C" {
void init_filter_query (void) __attribute__ ((constructor));
void fini_filter_query (void) __attribute__ ((destructor));
}

/** @brief tensor-filter-subplugin concrete class for Query */
class query_subplugin final : public tensor_filter_subplugin
{
  private:
    static const char *name_;
    static query_subplugin *registered_;

    grpc_idl idl_;
    std::string host_;
    gint port_;

    bool parse_custom_prop (const char *custom_prop);

  public:
    static void init_filter_query ();
    static void fini_filter_query ();

    query_subplugin ();
    ~query_subplugin ();

    tensor_filter_subplugin &getEmptyInstance ();
    void configure_instance (const GstTensorFilterProperties *prop);
    void invoke (const GstTensorMemory *input, GstTensorMemory *output);
    void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
    int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
    int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *query_subplugin::name_ = "query";
query_subplugin *query_subplugin::registered_ = nullptr;

#define DEFAULT_IDL   "flatbuf"
#define DEFAULT_HOST  "localhost"
#define DEFAULT_PORT  55111

/**
 * @brief Constructor for query_subplugin.
 */
query_subplugin::query_subplugin ()
  : tensor_filter_subplugin ()
{
  idl_ = grpc_get_idl (DEFAULT_IDL);
  host_ = DEFAULT_HOST;
  port_ = DEFAULT_PORT;
}

/**
 * @brief Method to cleanup query subplugin.
 */
query_subplugin::~query_subplugin ()
{
  /** TODO */
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
query_subplugin::getEmptyInstance ()
{
  return *(new query_subplugin ());
}

/**
 * @brief Internal method to parse custom options.
 */
bool
query_subplugin::parse_custom_prop (const char *custom_prop)
{
  if (!custom_prop) {
    /* no custom properties */
    return true;
  }

  gchar **options = g_strsplit (custom_prop, ",", -1);

  /* fall back to default values when invalid values are provided */
  for (guint op = 0; op < g_strv_length (options); ++op) {
    gchar **option = g_strsplit (options[op], ":", -1);

    if (g_strv_length (option) > 1) {
      g_strstrip (option[0]);
      g_strstrip (option[1]);

      const gchar * key = option[0];
      const gchar * val = option[1];

      if (g_ascii_strcasecmp (key, "idl") == 0) {
        grpc_idl idl = grpc_get_idl (val);
        if (idl != GRPC_IDL_NONE)
          idl_ = idl;
        else
          ml_logw ("Invalid 'idl' property provided, falling back to '%s'", DEFAULT_IDL);
      } else if (g_ascii_strcasecmp (key, "host") == 0) {
        if (g_strcmp0 (val, "localhost") == 0 || g_hostname_is_ip_address (val))
          host_ = val;
        else
          ml_logw ("Invalid 'host' property provided, falling back to '%s'", DEFAULT_HOST);
      } else if (g_ascii_strcasecmp (key, "port") == 0) {
        gchar * endptr = nullptr;
        gint64 out;

        errno = 0;
        out = g_ascii_strtoll (val, &endptr, 10);
        if (errno == 0 && val != endptr && out > 0 && out <= G_MAXUSHORT)
          port_ = (gint) out;
        else
          ml_logw ("Invalid 'port' property provided, falling back to '%d'", DEFAULT_PORT);
      }
    }

    g_strfreev (option);
  }

  g_strfreev (options);

  return true;
}

/**
 * @brief Method to prepare/configure Query instance.
 */
void
query_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (!parse_custom_prop (prop->custom_properties)) {
    throw std::invalid_argument ("Failed to configure options from custom properties");
  }
}

/**
 * @brief Method to execute the model.
 */
void
query_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  /** TODO */
}

/**
 * @brief Method to get the information of Query subplugin.
 */
void
query_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  /** TODO */
}

/**
 * @brief Method to get the model information.
 */
int
query_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  /** TODO */
  return 0;
}

/**
 * @brief Method to handle events.
 */
int
query_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

/**
 * @brief Initialize this object for tensor_filter subplugin runtime register
 */
void
query_subplugin::init_filter_query (void)
{
  registered_ = tensor_filter_subplugin::register_subplugin<query_subplugin> ();
}

/**
 * @brief Destruct the subplugin
 */
void
query_subplugin::fini_filter_query (void)
{
  assert (registered_ != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registered_);
}

/**
 * @brief Register the sub-plugin for Query.
 */
void init_filter_query ()
{
  query_subplugin::init_filter_query ();
}

/**
 * @brief Destruct the sub-plugin for Query.
 */
void fini_filter_query ()
{
  query_subplugin::fini_filter_query ();
}

} /* namespace nnstreamer::tensor_filter_query */
} /* namespace nnstreamer */
