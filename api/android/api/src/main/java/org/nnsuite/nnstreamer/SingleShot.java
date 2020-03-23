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

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

import java.io.File;

/**
 * Provides interfaces to invoke a neural network model with a single instance of input data.<br>
 * This function is a syntactic sugar of NNStreamer Pipeline API with simplified features;
 * thus, users are supposed to use NNStreamer Pipeline API directly if they want more advanced features.<br>
 * The user is expected to preprocess the input data for the given neural network model.<br>
 * <br>
 * {@link SingleShot} allows the following operations with NNStreamer:<br>
 * - Open a machine learning model.<br>
 * - Interfaces to enter a single instance of input data to the opened model.<br>
 * - Utility functions to get the information of opened model.<br>
 */
public final class SingleShot implements AutoCloseable {
    private long mHandle = 0;

    private native long nativeOpen(String[] models, TensorsInfo in, TensorsInfo out, int fw, String option);
    private native void nativeClose(long handle);
    private native TensorsData nativeInvoke(long handle, TensorsData in);
    private native TensorsInfo nativeGetInputInfo(long handle);
    private native TensorsInfo nativeGetOutputInfo(long handle);
    private native boolean nativeSetProperty(long handle, String name, String value);
    private native String nativeGetProperty(long handle, String name);
    private native boolean nativeSetInputInfo(long handle, TensorsInfo in);
    private native boolean nativeSetTimeout(long handle, int timeout);

    /**
     * Creates a new {@link SingleShot} instance with the given model for TensorFlow Lite.
     * If the model has flexible data dimensions, the pipeline will not be constructed and this will make an exception.
     *
     * @param model The path to the neural network model file
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public SingleShot(@NonNull File model) {
        this(model, null, null);
    }

    /**
     * Creates a new {@link SingleShot} instance with the given model for TensorFlow Lite.
     * The input and output tensors information are required if the given model has flexible data dimensions,
     * where the information MUST be given before executing the model.
     * However, once it's given, the dimension cannot be changed for the given model handle.
     * You may set null if it's not required.
     *
     * @param model The {@link File} object to the neural network model file
     * @param in    The input tensors information
     * @param out   The output tensors information
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public SingleShot(@NonNull File model, @Nullable TensorsInfo in, @Nullable TensorsInfo out) {
        this(new File[]{model}, in, out, NNStreamer.NNFWType.TENSORFLOW_LITE, null);
    }

    /**
     * Creates a new {@link SingleShot} instance with the given files and custom option.
     *
     * Unlike other constructors, this handles multiple files and custom option string
     * when the neural network requires various options and model files.
     *
     * @param models The array of {@link File} objects to the neural network model files
     * @param in     The input tensors information
     * @param out    The output tensors information
     * @param fw     The neural network framework
     * @param option The custom option string to open the neural network
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to construct the pipeline
     *
     * @see NNStreamer#isAvailable(NNStreamer.NNFWType)
     */
    public SingleShot(@NonNull File[] models, @Nullable TensorsInfo in, @Nullable TensorsInfo out,
                      NNStreamer.NNFWType fw, @Nullable String option) {
        if (models == null) {
            throw new IllegalArgumentException("Given model is invalid");
        }

        if (!NNStreamer.isAvailable(fw)) {
            throw new IllegalStateException("Given framework is not available");
        }

        String[] path = new String[models.length];
        int index = 0;

        for (File model : models) {
            if (model == null || !model.exists()) {
                throw new IllegalArgumentException("Given model is invalid");
            }

            path[index++] = model.getAbsolutePath();
        }

        mHandle = nativeOpen(path, in, out, fw.ordinal(), option);
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to construct the pipeline");
        }
    }

    /**
     * Invokes the model with the given input data.
     * If the model has flexible input data dimensions, input information for this
     * run of the model can be passed. This changes the currently set input information
     * for this instance of the model. The corresponding output information can be
     * extracted.
     *
     * Note that this has a default timeout of 3 seconds.
     * If an application wants to change the time to wait for an output,
     * set the timeout using {@link #setTimeout(int)}.
     *
     * @param in The input data to be inferred (a single frame, tensor/tensors)
     *
     * @return The output data (a single frame, tensor/tensors)
     *
     * @throws IllegalStateException if failed to invoke the model
     * @throws IllegalArgumentException if given param is null
     */
    public TensorsData invoke(@NonNull TensorsData in) {
        checkPipelineHandle();

        if (in == null) {
            throw new IllegalArgumentException("Given input data is null");
        }

        TensorsData out = nativeInvoke(mHandle, in);
        if (out == null) {
            throw new IllegalStateException("Failed to invoke the model");
        }

        return out;
    }

    /**
     * Gets the information (tensor dimension, type, name and so on) of required input data for the given model.
     *
     * @return The tensors information
     *
     * @throws IllegalStateException if failed to get the input information
     */
    public TensorsInfo getInputInfo() {
        checkPipelineHandle();

        TensorsInfo info = nativeGetInputInfo(mHandle);
        if (info == null) {
            throw new IllegalStateException("Failed to get the input information");
        }

        return info;
    }

    /**
     * Gets the information (tensor dimension, type, name and so on) of output data for the given model.
     *
     * @return The tensors information
     *
     * @throws IllegalStateException if failed to get the output information
     */
    public TensorsInfo getOutputInfo() {
        checkPipelineHandle();

        TensorsInfo info = nativeGetOutputInfo(mHandle);
        if (info == null) {
            throw new IllegalStateException("Failed to get the output information");
        }

        return info;
    }

    /**
     * Sets the property value for the given model.
     * Note that a model/framework may not support to change the property after opening the model.
     *
     * @param name  The property name
     * @param value The property value
     *
     * @throws IllegalArgumentException if given param is invalid
     */
    public void setValue(@NonNull String name, @NonNull String value) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Given property name is invalid");
        }

        if (value == null) {
            throw new IllegalArgumentException("Given property value is invalid");
        }

        if (!nativeSetProperty(mHandle, name, value)) {
            throw new IllegalArgumentException("Failed to set the property");
        }
    }

    /**
     * Gets the property value for the given model.
     *
     * @param name The property name
     *
     * @return The property value
     *
     * @throws IllegalArgumentException if given param is invalid
     */
    public String getValue(@NonNull String name) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Given property name is invalid");
        }

        String value = nativeGetProperty(mHandle, name);

        if (value == null) {
            throw new IllegalArgumentException("Failed to get the property");
        }

        return value;
    }

    /**
     * Sets the maximum amount of time to wait for an output, in milliseconds.
     *
     * @param timeout The time to wait for an output
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to set the timeout
     */
    public void setTimeout(int timeout) {
        checkPipelineHandle();

        if (timeout <= 0) {
            throw new IllegalArgumentException("Given timeout is invalid");
        }

        if (!nativeSetTimeout(mHandle, timeout)) {
            throw new IllegalStateException("Failed to set the timeout");
        }
    }

    /**
     * Sets the information (tensor dimension, type, name and so on) of input data for the given model.
     * Updates the output information for the model internally.
     *
     * @param in The input tensors information
     *
     * @throws IllegalStateException if failed to set the input information
     * @throws IllegalArgumentException if given param is null
     */
    public void setInputInfo(@NonNull TensorsInfo in) {
        checkPipelineHandle();

        if (in == null) {
            throw new IllegalArgumentException("Given input info is null");
        }

        if (!nativeSetInputInfo(mHandle, in)) {
            throw new IllegalStateException("Failed to set input tensor info");
        }
    }

    /**
     * Internal method to check native handle.
     *
     * @throws IllegalStateException if the pipeline is not constructed
     */
    private void checkPipelineHandle() {
        if (mHandle == 0) {
            throw new IllegalStateException("The pipeline is not constructed");
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    @Override
    public void close() {
        if (mHandle != 0) {
            nativeClose(mHandle);
            mHandle = 0;
        }
    }
}
