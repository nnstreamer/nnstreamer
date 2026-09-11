/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd.
 *
 * @file        gsttensor_compositor.c
 * @date        26 Oct 2023
 * @brief       GStreamer element to composite multiple tensor streams
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items.
 */


/**
 * SECTION:element-tensor_compositor
 *
 * tensor_compositor is a GStreamer element to composite tensor streams.
 *
 * tensor_compositor can have multiple sink pads of other/tensors,
 * and has a single source pad of other/tensors.
 * It is similar to the compositor element, which composites video streams.
 * And intuitively, the input tensor streams can be regarded as video streams
 * to be composited by the element.
 *
 * If the two input tensors are to be composited disjointedly without any
 * padding or filling, it can be done by tensor_merge, too.
 *
 * The input streams may have different dimensions or formats; however,
 * the input streams should have the same number of tensors (num_tensors),
 * and the same types for every matching tensor (e.g., uint8, float32, ...).
 *
 * The output is always other/tensors,format=static.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 videotestsrc ! videoconvert ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! cmp.sink_0
 *     videotestsrc ! videoconvert ! video/x-raw,format=RGB,width=1920,height=1080 ! tensor_converter ! cmp.sink_1
         tensor_compositor name=cmp num_tensors=1 types=uint8 dimensions=3:1920:1080 sink_0::z_order=1 sink_1::z_order=2 sink0::pos=0:640:480 sink1::pos=0:0:0 merge_logic=add ! tensor_decoder mode=direct_video ! videoconvert ! autovideosink
 * ]|
 * </refsects>
 */
