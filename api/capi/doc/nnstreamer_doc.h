/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */
/**
 * @file nnstreamer_doc.h
 * @date 06 March 2019
 * @brief Tizen C-API Declaration for Tizen SDK
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNSTREAMER_DOC_H__
#define __TIZEN_MACHINELEARNING_NNSTREAMER_DOC_H__

/**
 * @ingroup CAPI_ML_FRAMEWORK
 * @defgroup CAPI_ML_NNSTREAMER NNStreamer
 * @brief NNStreamer allows to construct and execute a GStreamer pipeline including neural networks.\n
 * NNStreamer is a set of GStreamer plugins that allow GStreamer developers to adopt neural network models easily and efficiently and neural network developers to manage stream pipelines and their filters easily and efficiently.\n
 * https://github.com/nnsuite/nnstreamer/\n
 * \n
 * Main objectives of NNStreamer include:\n
 * * Provide neural network framework connectivity (e.g., tensorflow, caffe) for GStreamer streams.\n
 *     * **Efficient Streaming for AI Projects**: Apply efficient and flexible stream pipeline to neural networks.\n
 *     * **Intelligent Media Filters!**: Use a neural network model as a media filter / converter.\n
 *     * **Composite Models!**: Multiple neural network models in a single stream pipeline instance.\n
 *     * **Multi Modal Intelligence!**: Multiple sources and stream paths for neural network models.\n
 * * Provide easy methods to construct media streams with neural network models using the de-facto-standard media stream framework, **GStreamer**.\n
 *     * GStreamer users: use neural network models as if they are yet another media filters.\n
 *     * Neural network developers: manage media streams easily and efficiently.\n
 * \n
 * There are following sub groups proposed:\n
 * * Pipeline: construct and control an NNStreamer pipeline.\n
 * * Single: invoke a neural network model with a single instance of input data.\n
 *
 * @defgroup CAPI_ML_NNSTREAMER_PIPELINE_MODULE NNStreamer Pipeline
 * @ingroup  CAPI_ML_NNSTREAMER
 * @brief The NNStreamer function provides interfaces to create and execute stream pipelines with neural networks and sensors.
 * @section CAPI_ML_NNSTREAMER_PIPELINE_HEADER Required Header
 *   \#include <nnstreamer/nnstreamer.h>\n
 *
 * @section CAPI_ML_NNSTREAMER_PIPELINE_OVERVIEW Overview
 * The NNStreamer function provides interfaces to create and execute stream pipelines with neural networks and sensors.
 *
 *  This function allows the following operations with NNStreamer:
 * - Create a stream pipeline with NNStreamer plugins, GStreamer plugins, and sensor/camera/mic inputs.
 * - Interfaces to push data to the pipeline from the application.
 * - Interfaces to pull data from the pipeline to the application.
 * - Interfaces to start/stop/destroy the pipeline.
 * - Interfaces to control switches and valves in the pipeline.
 * - Utility functions to handle data from the pipeline.
 *
 *  Note that this function set is supposed to be thread-safe.
 *
 * @section CAPI_ML_NNSTREAMER_PIPELINE_FEATURE Related Features
 * This function is related with the following features:\n
 *  - http://tizen.org/feature/nnstreamer.pipeline\n
 *
 * It is recommended to probe features in your application for reliability.\n
 * You can check if a device supports the related features for this function by using
 * @ref CAPI_SYSTEM_SYSTEM_INFO_MODULE, thereby controlling the procedure of
 * your application.\n
 * To ensure your application is only running on the device with specific
 * features, please define the features in your manifest file using the manifest
 * editor in the SDK.\n
 * For example, your application accesses to the camera device,
 * then you have to add 'http://tizen.org/privilege/camera' into the manifest of your application.\n
 * More details on featuring your application can be found from
 * <a href="https://developer.tizen.org/development/tizen-studio/native-tools/configuring-your-app/manifest-text-editor#feature">
 *    <b>Feature Element</b>.
 * </a>
 *
 * @defgroup CAPI_ML_NNSTREAMER_SINGLE_MODULE NNStreamer Single Shot
 * @ingroup  CAPI_ML_NNSTREAMER
 * @brief The NNStreamer Single API provides interfaces to invoke a neural network model with a single instance of input data.
 * @section CAPI_ML_NNSTREAMER_SINGLE_HEADER Required Header
 *   \#include <nnstreamer/nnstreamer-single.h>\n
 *
 * @section CAPI_ML_NNSTREAMER_SINGLE_OVERVIEW Overview
 * The NNStreamer Single API provides interfaces to invoke a neural network model with a single instance of input data.
 * This function is a syntactic sugar of NNStreamer Pipeline API with simplified features; thus, users are supposed to use NNStreamer Pipeline API directly if they want more advanced features.
 * The user is expected to preprocess the input data for the given neural network model.
 *
 * This function allows the following operations with NNStreamer:
 * - Open a machine learning model with various mechanisms.
 * - Close the model.
 * - Interfaces to enter a single instance of input data to the opened model.
 * - Utility functions to get the information of opened model.
 *
 * Note that this function set is supposed to be thread-safe.
 *
 * @section CAPI_ML_NNSTREAMER_SINGLE_FEATURE Related Features
 * This function is related with the following features:\n
 *  - http://tizen.org/feature/nnstreamer.single\n
 *
 * It is recommended to probe features in your application for reliability.\n
 * You can check if a device supports the related features for this function by using
 * @ref CAPI_SYSTEM_SYSTEM_INFO_MODULE, thereby controlling the procedure of
 * your application.\n
 * To ensure your application is only running on the device with specific
 * features, please define the features in your manifest file using the manifest
 * editor in the SDK.\n
 * For example, your application accesses to the camera device,
 * then you have to add 'http://tizen.org/privilege/camera' into the manifest of your application.\n
 * More details on featuring your application can be found from
 * <a href="https://developer.tizen.org/development/tizen-studio/native-tools/configuring-your-app/manifest-text-editor#feature">
 *    <b>Feature Element</b>.
 * </a>
 */


#endif /* __TIZEN_MACHINELEARNING_NNSTREAMER_DOC_H__ */
