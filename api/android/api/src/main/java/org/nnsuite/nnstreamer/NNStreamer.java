/*
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
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
 * See <a href="https://github.com/nnsuite/nnstreamer">https://github.com/nnsuite/nnstreamer</a> for the details.
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
     * Gets the version string of GStreamer and NNStreamer.
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
