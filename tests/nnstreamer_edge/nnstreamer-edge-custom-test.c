/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (C) 2024 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   nnstreamer-edge-custom-test.c
 * @date   30 Aug 2024
 * @brief  NNStreamer-edge custom connection for test.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "nnstreamer-edge.h"
#include "nnstreamer-edge-custom.h"
#include "nnstreamer_log.h"
#include "nnstreamer_util.h"
#include <glib.h>

#define SAFE_FREE(p) do { if (p) { free (p); (p) = NULL; } } while (0)

typedef struct
{
  int is_connected;
  char *peer_address;
  nns_edge_event_cb event_cb;
  void *user_data;
} nns_edge_custom_test_s;

/**
 * @brief Release the private handle of the test custom connection.
 */
static int
nns_edge_custom_close (void *priv)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;

  SAFE_FREE (custom_h->peer_address);
  SAFE_FREE (custom_h);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Return the description string of this custom connection.
 */
static const char *
nns_edge_custom_get_description (void)
{
  return "custom";
}

/**
 * @brief Allocate the private handle of the test custom connection.
 */
static int
nns_edge_custom_create (void **priv)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  custom_h = (nns_edge_custom_test_s *) calloc (1, sizeof (nns_edge_custom_test_s));
  if (!custom_h) {
    nns_loge ("Failed to allocate memory for edge custom handle.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  *priv = custom_h;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Start the custom connection, resetting it to the unconnected state.
 */
static int
nns_edge_custom_start (void *priv)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;
  custom_h->is_connected = 0;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Stop the custom connection and clear its connected flag.
 */
static int
nns_edge_custom_stop (void *priv)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;
  custom_h->is_connected = 0;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Mark the connection as connected and push one dummy data event.
 */
static int
nns_edge_custom_connect (void *priv)
{
  nns_edge_custom_test_s *custom_h;
  nns_edge_data_h data_h;
  gchar *raw_data;
  int ret = NNS_EDGE_ERROR_INVALID_PARAMETER;

  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return ret;
  }
  custom_h = (nns_edge_custom_test_s *) priv;
  custom_h->is_connected = 1;

  /* Push dummy buffer to launch GstBaseSrc */
  ret = nns_edge_data_create (&data_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to create edge data handle.");
    return ret;
  }

  raw_data = g_strdup ("Dummy data");
  ret = nns_edge_data_add (data_h, raw_data, strlen (raw_data) + 1, g_free);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to add edge data.");
    g_free (raw_data);
    goto done;
  }
  ret = nns_edge_event_invoke_callback (custom_h->event_cb, custom_h->user_data,
      NNS_EDGE_EVENT_NEW_DATA_RECEIVED, data_h, sizeof (nns_edge_data_h), NULL);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to invoke edge event.");
  }

done:
  nns_edge_data_destroy (data_h);

  return ret;
}

/**
 * @brief Subscription is not supported by this test custom connection.
 */
static int
nns_edge_custom_subscribe (void *priv)
{
  UNUSED (priv);
  return NNS_EDGE_ERROR_NOT_SUPPORTED;
}

/**
 * @brief Report whether the test custom connection is connected.
 */
static int
nns_edge_custom_is_connected (void *priv)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;

  if (custom_h->is_connected == 1)
    return NNS_EDGE_ERROR_NONE;

  return NNS_EDGE_ERROR_CONNECTION_FAILURE;
}

/**
 * @brief Store the event callback and its user data in the handle.
 */
static int
nns_edge_custom_set_event_cb (void *priv, nns_edge_event_cb cb, void *user_data)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv) {
    nns_loge ("Invalid param, handle should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;

  custom_h->event_cb = cb;
  custom_h->user_data = user_data;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Discard the given data after validating the parameters.
 */
static int
nns_edge_custom_send_data (void *priv, nns_edge_data_h data_h)
{
  if (!priv || !data_h) {
    nns_loge ("Invalid param, handle or data should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Store the PEER_ADDRESS value; every other key is ignored.
 */
static int
nns_edge_custom_set_info (void *priv, const char *key, const char *value)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv || !key || !value) {
    nns_loge ("Invalid param, handle, key or value should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;

  if (strcasecmp (key, "PEER_ADDRESS") == 0) {
    SAFE_FREE (custom_h->peer_address);
    custom_h->peer_address = g_strdup (value);
    return NNS_EDGE_ERROR_NONE;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Return a newly allocated copy of the stored PEER_ADDRESS value.
 */
static int
nns_edge_custom_get_info (void *priv, const char *key, char **value)
{
  nns_edge_custom_test_s *custom_h;
  if (!priv || !key || !value) {
    nns_loge ("Invalid param, handle, key or value should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  custom_h = (nns_edge_custom_test_s *) priv;

  if (strcasecmp (key, "PEER_ADDRESS") == 0) {
    *value = g_strdup (custom_h->peer_address);
    return NNS_EDGE_ERROR_NONE;
  }

  nns_loge ("The key '%s' is not supported.", key);
  return NNS_EDGE_ERROR_INVALID_PARAMETER;
}

nns_edge_custom_s edge_custom_h = {
  .nns_edge_custom_get_description = nns_edge_custom_get_description,
  .nns_edge_custom_create = nns_edge_custom_create,
  .nns_edge_custom_close = nns_edge_custom_close,
  .nns_edge_custom_start = nns_edge_custom_start,
  .nns_edge_custom_stop = nns_edge_custom_stop,
  .nns_edge_custom_connect = nns_edge_custom_connect,
  .nns_edge_custom_subscribe = nns_edge_custom_subscribe,
  .nns_edge_custom_is_connected = nns_edge_custom_is_connected,
  .nns_edge_custom_set_event_cb = nns_edge_custom_set_event_cb,
  .nns_edge_custom_send_data = nns_edge_custom_send_data,
  .nns_edge_custom_set_info = nns_edge_custom_set_info,
  .nns_edge_custom_get_info = nns_edge_custom_get_info
};

/**
 * @brief Return the callback table of this test custom connection.
 */
const nns_edge_custom_s *
nns_edge_custom_get_instance (void)
{
  return &edge_custom_h;
}
