/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_common.h
 * @date    01 Aug 2022
 * @brief   Common functions for edge sink and src
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_EDGE_H__
#define __GST_EDGE_H__

G_BEGIN_DECLS
#ifndef UNUSED
#define UNUSED(expr) do { (void)(sizeof(x), 0); } while (0)
#endif /* UNUSED */
#ifndef GST_EDGE_PACKAGE
#define GST_EDGE_PACKAGE "GStreamer Edge Plugins"
#endif /* GST_EDGE_PACKAGE */
#define GST_EDGE_ELEM_NAME_SINK "edgesink"
#define GST_EDGE_ELEM_NAME_SRC "edgesrc"
    G_END_DECLS
#endif /* __GST_EDGE_H__ */
