/* SPDX-License-Identifier: LGPL-2.1-only */
/**

* @file        nnstreamer_flatbuf.h
* @date        11 Aug 2022
* @brief       Common contents for NNStreamer flatbuf-related subplugins.
* @see         https://github.com/nnstreamer/nnstreamer
* @author      MyungJoo Ham <myungjoo.ham@samsung.com>
* @bug         No known bugs except for NYI items
*
*/
#ifndef __NNSTREAMER_FLATBUF_H__
#define __NNSTREAMER_FLATBUF_H__

namespace nnstreamer
{
namespace flatbuf
{

/**
 * @brief Default static capability for flatbuffers
 * Flatbuf converter will convert this capability to other/tensor(s)
 * @todo Move this definition to proper header file
 */
#define GST_FLATBUF_TENSOR_CAP_DEFAULT \
    "other/flatbuf-tensor, " \
    "framerate = " GST_TENSOR_RATE_RANGE

}; /* Namespace flatbuf */
}; /* Namespace nnstreamer */

/**
 * @brief Default static capability for flexbuffers
 */
#define GST_FLEXBUF_CAP_DEFAULT "other/flexbuf"

#endif /* __NNSTREAMER_FLATBUF_H__ */
