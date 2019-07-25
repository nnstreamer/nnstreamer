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

package com.samsung.android.nnstreamer;

import android.content.Context;

import org.freedesktop.gstreamer.GStreamer;

/**
 * Defines the types and limits in NNStreamer.<br>
 * To use NNStreamer, an application should call {@link #initialize(Context)} with its context.<br>
 * <br>
 * NNStreamer is a set of GStreamer plugins that allow GStreamer developers to adopt neural network models easily and efficiently
 * and neural network developers to manage stream pipelines and their filters easily and efficiently.<br>
 * <br>
 * See <a href="https://github.com/nnsuite/nnstreamer">https://github.com/nnsuite/nnstreamer</a> for the details.
 */
public final class NNStreamer {
    /**
     * The maximum rank that NNStreamer supports.
     */
    public static final int TENSOR_RANK_LIMIT = 4;

    /**
     * The maximum number of tensor instances that tensors may have.
     */
    public static final int TENSOR_SIZE_LIMIT = 16;

    /**
     * The data type of tensor in NNStreamer: Integer 32bit.
     */
    public static final int TENSOR_TYPE_INT32 = 0;

    /**
     * The data type of tensor in NNStreamer: Unsigned integer 32bit.
     */
    public static final int TENSOR_TYPE_UINT32 = 1;

    /**
     * The data type of tensor in NNStreamer: Integer 16bit.
     */
    public static final int TENSOR_TYPE_INT16 = 2;

    /**
     * The data type of tensor in NNStreamer: Unsigned integer 16bit.
     */
    public static final int TENSOR_TYPE_UINT16 = 3;

    /**
     * The data type of tensor in NNStreamer: Integer 8bit.
     */
    public static final int TENSOR_TYPE_INT8 = 4;

    /**
     * The data type of tensor in NNStreamer: Unsigned integer 8bit.
     */
    public static final int TENSOR_TYPE_UINT8 = 5;

    /**
     * The data type of tensor in NNStreamer: Float 64bit.
     */
    public static final int TENSOR_TYPE_FLOAT64 = 6;

    /**
     * The data type of tensor in NNStreamer: Float 32bit.
     */
    public static final int TENSOR_TYPE_FLOAT32 = 7;

    /**
     * The data type of tensor in NNStreamer: Integer 64bit.
     */
    public static final int TENSOR_TYPE_INT64 = 8;

    /**
     * The data type of tensor in NNStreamer: Unsigned integer 64bit.
     */
    public static final int TENSOR_TYPE_UINT64 = 9;

    /**
     * Unknown data type of tensor in NNStreamer.
     */
    public static final int TENSOR_TYPE_UNKNOWN = 10;

    /**
     * The state of pipeline: Unknown state.
     */
    public static final int PIPELINE_STATE_UNKNOWN = 0;

    /**
     * The state of pipeline: Initial state of the pipeline.
     */
    public static final int PIPELINE_STATE_NULL = 1;

    /**
     * The state of pipeline: The pipeline is ready to go to PAUSED.
     */
    public static final int PIPELINE_STATE_READY = 2;

    /**
     * The state of pipeline: The pipeline is stopped, ready to accept and process data.
     */
    public static final int PIPELINE_STATE_PAUSED = 3;

    /**
     * The state of pipeline: The pipeline is started and the data is flowing.
     */
    public static final int PIPELINE_STATE_PLAYING = 4;

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
            return nativeInitialize(context);
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Gets the version string of GStreamer and NNStreamer.
     *
     * @return The version string
     */
    public static String getVersion() {
        return nativeGetVersion();
    }
}
