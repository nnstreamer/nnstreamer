/*
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Library General Public License for more details.
 */

package org.nnsuite.nnstreamer;

import android.content.Context;

import org.freedesktop.gstreamer.GStreamer;

/**
 * Defines the types and limits in NNStreamer.<br>
 * To use NNStreamer, an application should call {@link #initialize(Context)} with its context.<br>
 * <br>
 * NNStreamer is a set of GStreamer plugins that allow GStreamer developers to adopt neural network models easily and efficiently
 * and neural network developers to manage stream pipelines and their filters easily and efficiently.<br>
 * <br>
 * Note that, to open a machine learning model in the storage,
 * the permission <code>Manifest.permission.READ_EXTERNAL_STORAGE</code> is required before constructing the pipeline.
 * <br>
 * See <a href="https://github.com/nnstreamer/nnstreamer">https://github.com/nnstreamer/nnstreamer</a> for the details.
 */
public final class NNStreamer {
    /**
     * The maximum rank that NNStreamer supports.
     */
    public static final int TENSOR_RANK_LIMIT = 4;

    /**
     * The maximum number of tensor that {@link TensorsData} instance may have.
     */
    public static final int TENSOR_SIZE_LIMIT = 16;

    /**
     * The enumeration for supported frameworks in NNStreamer.
     */
    public enum NNFWType {
        /**
         * TensorFlow Lite<br>
         * <br>
         * <a href="https://www.tensorflow.org/lite">TensorFlow Lite</a> is an open source
         * deep learning framework for on-device inference.<br>
         */
        TENSORFLOW_LITE,
        /**
         * SNAP (Samsung Neural Acceleration Platform)<br>
         * <br>
         * Supports <a href="https://developer.samsung.com/neural">Samsung Neural SDK</a>
         * (Version 1.0, run only on Samsung devices)<br>
         * To construct a pipeline with SNAP, developer should set the custom option string
         * to specify the neural network and data format.<br>
         * <br>
         * Custom options<br>
         * - ModelFWType: the type of model (TensorFlow/Caffe)<br>
         * - ExecutionDataType: the execution data type for SNAP (default float32)<br>
         * - ComputingUnit: the computing unit to execute the model (default CPU)<br>
         * - CpuThreadCount: the number of CPU threads to be executed (optional, default 4 if ComputingUnit is CPU)<br>
         * - GpuCacheSource: the absolute path to GPU Kernel caching (mandatory if ComputingUnit is GPU)<br>
         */
        SNAP,
        /**
         * Unknown framework (usually error)
         */
        UNKNOWN
    }

    /**
     * The enumeration for possible data type of tensor in NNStreamer.
     */
    public enum TensorType {
        /** Integer 32bit */ INT32,
        /** Unsigned integer 32bit */ UINT32,
        /** Integer 16bit */ INT16,
        /** Unsigned integer 16bit */ UINT16,
        /** Integer 8bit */ INT8,
        /** Unsigned integer 8bit */ UINT8,
        /** Float 64bit */ FLOAT64,
        /** Float 32bit */ FLOAT32,
        /** Integer 64bit */ INT64,
        /** Unsigned integer 64bit */ UINT64,
        /** Unknown data type (usually error) */ UNKNOWN
    }

    private static native boolean nativeInitialize(Context context);
    private static native boolean nativeCheckAvailability(int fw);
    private static native String nativeGetVersion();

    /**
     * Initializes GStreamer and NNStreamer, registering the plugins and loading necessary libraries.
     *
     * @param context The application context
     *
     * @return true if successfully initialized
     */
    public static boolean initialize(Context context) {
        try {
            System.loadLibrary("nnstreamer-native");
            GStreamer.init(context);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return nativeInitialize(context);
    }

    /**
     * Checks the neural network framework is available.
     *
     * @param fw The neural network framework
     *
     * @return true if the neural network framework is available
     */
    public static boolean isAvailable(NNFWType fw) {
        return nativeCheckAvailability(fw.ordinal());
    }

    /**
     * Gets the version string of NNStreamer.
     *
     * @return The version string
     */
    public static String getVersion() {
        return nativeGetVersion();
    }

    /**
     * Private constructor to prevent the instantiation.
     */
    private NNStreamer() {}
}
