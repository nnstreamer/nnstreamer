/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file   nnstreamer-edge-custom-test-api.h
 * @brief  Shared definitions for the nnstreamer-edge custom test plugin.
 */
#ifndef __NNSTREAMER_EDGE_CUSTOM_TEST_API_H__
#define __NNSTREAMER_EDGE_CUSTOM_TEST_API_H__

/**
 * @brief Payload generation modes for the custom edge test backend.
 */
typedef enum
{
  EDGE_CUSTOM_PAYLOAD_MODE_DEFAULT = 0,
  EDGE_CUSTOM_PAYLOAD_MODE_STATIC = 1,
} edge_custom_payload_mode_e;

#define EDGE_CUSTOM_TEST_STATIC_PAYLOAD_SIZE 64U

#define EDGE_CUSTOM_TEST_LIB "libnnstreamer-edge-custom-test.so"

#endif /* __NNSTREAMER_EDGE_CUSTOM_TEST_API_H__ */
