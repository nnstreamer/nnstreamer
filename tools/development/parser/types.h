/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * This is imported from GStreamer and altered to parse GST-Pipeline
 *
 * @file  types.h
 * @date  12 Nov 2020
 * @brief GST-Parser's types.h modified for gst2pbtxt parser (nnstreamer parser)
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __GST_PARSE_TYPES_H__
#define __GST_PARSE_TYPES_H__

#include <glib-object.h>

/** @brief Simplified GST-Element */
typedef struct {
  gchar *element;
  gchar *name;
  GSList *properties; /**< List of key-value pairs (_Property), added for gst-pbtxt, except for name=.... */
} _Element;

extern _Element *
nnstparser_element_make (const gchar *element, const gchar *name);

/** @brief Simplified GObject Property for GST Element */
typedef struct {
  gchar *name;
  gchar *value;
} _Property;

/** @brief pipeline element/pad reference */
typedef struct {
  _Element *element;
  gchar *name;
  GSList *pads; /**< Unlike original, it's list of string pad-names */
} reference_t;

/** @brief pad-to-pad linkage relation */
typedef struct {
  reference_t src;
  reference_t sink;
  gchar *caps; /**< In this app, caps is generally unidentifiable. simply copy the caps string! */
  gboolean all_pads;
} link_t;

/** @brief Chain of elements */
typedef struct {
  GSList *elements; /**< Originally its data is "GstElement". It's now _Element */
  reference_t first;
  reference_t last;
} chain_t;

/**
 * @brief A dummy created for gst2pbtxt
 */
typedef struct {
  GList *missing_elements;
} _ParseContext;

/**
 * @brief A dummy created for gst2pbtxt
 */
typedef enum
{
  __PARSE_FLAG_NONE = 0,
  __PARSE_FLAG_FATAL_ERRORS = (1 << 0),
  __PARSE_FLAG_NO_SINGLE_ELEMENT_BINS = (1 << 1),
  __PARSE_FLAG_PLACE_IN_BIN = (1 << 2)
} _ParseFlags;

typedef struct _graph_t graph_t;
/** @brief The pipeline graph */
struct _graph_t {
  chain_t *chain; /* links are supposed to be done now */
  GSList *links;
  GError **error;
  _ParseContext *ctx; /* may be NULL */
  _ParseFlags flags;
};


/**
 * Memory checking. Should probably be done with gsttrace stuff, but that
 * doesn't really work.
 * This is not safe from reentrance issues, but that doesn't matter as long as
 * we lock a mutex before parsing anyway.
 *
 * FIXME: Disable this for now for the above reasons
 */
#if 0
#ifdef GST_DEBUG_ENABLED
#  define __GST_PARSE_TRACE
#endif
#endif

#ifdef __GST_PARSE_TRACE
G_GNUC_INTERNAL  gchar  *__gst_parse_strdup (gchar *org);
G_GNUC_INTERNAL  void	__gst_parse_strfree (gchar *str);
G_GNUC_INTERNAL  link_t *__gst_parse_link_new (void);
G_GNUC_INTERNAL  void	__gst_parse_link_free (link_t *data);
G_GNUC_INTERNAL  chain_t *__gst_parse_chain_new (void);
G_GNUC_INTERNAL  void	__gst_parse_chain_free (chain_t *data);
#  define gst_parse_strdup __gst_parse_strdup
#  define gst_parse_strfree __gst_parse_strfree
#  define gst_parse_link_new __gst_parse_link_new
#  define gst_parse_link_free __gst_parse_link_free
#  define gst_parse_chain_new __gst_parse_chain_new
#  define gst_parse_chain_free __gst_parse_chain_free
#else /* __GST_PARSE_TRACE */
#  define gst_parse_strdup g_strdup
#  define gst_parse_strfree g_free
#  define gst_parse_link_new() g_slice_new0 (link_t)
#  define gst_parse_link_free(l) g_slice_free (link_t, l)
#  define gst_parse_chain_new() g_slice_new0 (chain_t)
#  define gst_parse_chain_free(c) g_slice_free (chain_t, c)
#endif /* __GST_PARSE_TRACE */

/** @brief GST Parser's internal function imported */
static inline void
gst_parse_unescape (gchar *str)
{
  gchar *walk;
  gboolean in_quotes;

  g_return_if_fail (str != NULL);

  walk = str;
  in_quotes = FALSE;

  GST_DEBUG ("unescaping %s", str);

  while (*walk) {
    if (*walk == '\\' && !in_quotes) {
      walk++;
      /* make sure we don't read beyond the end of the string */
      if (*walk == '\0')
        break;
    } else if (*walk == '"' && (!in_quotes || (in_quotes
                && (*(walk - 1) != '\\')))) {
      /** don't unescape inside quotes and don't switch
       * state with escaped quoted inside quotes */
      in_quotes = !in_quotes;
    }
    *str = *walk;
    str++;
    walk++;
  }
  *str = '\0';
}

GQuark gst2pbtxt_parse_error_quark (void);
#define GST2PBTXT_PARSE_ERROR gst2pbtxt_parse_error_quark ()

typedef enum
{
  GST2PBTXT_PARSE_ERROR_SYNTAX,
  GST2PBTXT_PARSE_ERROR_NO_SUCH_ELEMENT,
  GST2PBTXT_PARSE_ERROR_NO_SUCH_PROPERTY,
  GST2PBTXT_PARSE_ERROR_LINK,
  GST2PBTXT_PARSE_ERROR_COULD_NOT_SET_PROPERTY,
  GST2PBTXT_PARSE_ERROR_EMPTY_BIN,
  GST2PBTXT_PARSE_ERROR_EMPTY,
  GST2PBTXT_PARSE_ERROR_DELAYED_LINK
} Gst2PbtxtParseError;


G_GNUC_INTERNAL _Element *priv_gst_parse_launch (const gchar      * str,
                                                   GError          ** err,
                                                   _ParseContext  * ctx,
                                                   _ParseFlags      flags);

#endif /* __GST_PARSE_TYPES_H__ */
