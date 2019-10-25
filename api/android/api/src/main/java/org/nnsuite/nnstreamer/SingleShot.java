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

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

import java.io.File;

/**
 * Provides interfaces to invoke a neural network model with a single instance of input data.<br>
 * This function is a syntactic sugar of NNStreamer Pipeline API with simplified features;
 * thus, users are supposed to use NNStreamer Pipeline API directly if they want more advanced features.<br>
 * The user is expected to preprocess the input data for the given neural network model.<br>
 * <br>
 * <code>SingleShot</code> allows the following operations with NNStreamer:<br>
 * - Open a machine learning model.<br>
 * - Interfaces to enter a single instance of input data to the opened model.<br>
 * - Utility functions to get the information of opened model.<br>
 */
public final class SingleShot implements AutoCloseable {
    private long mHandle = 0;

    private native long nativeOpen(String model, TensorsInfo in, TensorsInfo out);
    private native void nativeClose(long handle);
    private native TensorsData nativeInvoke(long handle, TensorsData in, TensorsInfo inInfo);
    private native TensorsInfo nativeGetInputInfo(long handle);
    private native TensorsInfo nativeGetOutputInfo(long handle);
    private native boolean nativeSetInputInfo(long handle, TensorsInfo in);
    private native boolean nativeSetTimeout(long handle, int timeout);

    /**
     * Creates a new <code>SingleShot</code> instance with the given model.
     * If the model has flexible data dimensions, the pipeline will not be constructed and this will make an exception.
     *
     * @param model The path to the neural network model file
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public SingleShot(@NonNull File model) {
        this(model, null, null);
    }

    /**
     * Creates a new <code>SingleShot</code> instance with the given model.
     * The input and output tensors information are required if the given model has flexible data dimensions,
     * where the information MUST be given before executing the model.
     * However, once it's given, the dimension cannot be changed for the given model handle.
     * You may set null if it's not required.
     *
     * @param model The path to the neural network model file
     * @param in    The input tensors information
     * @param out   The output tensors information
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public SingleShot(@NonNull File model, @Nullable TensorsInfo in, @Nullable TensorsInfo out) {
        if (model == null || !model.exists()) {
            throw new IllegalArgumentException("The param model is invalid");
        }

        String path = model.getAbsolutePath();

        mHandle = nativeOpen(path, in, out);
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to construct the pipeline");
        }
    }

    /**
     * Invokes the model with the given input data.
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
        return invoke(in, null);
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
     * @param inData The input data to be inferred (a single frame, tensor/tensors)
     * @param inInfo The input tensors information
     *
     * @return The output data (a single frame, tensor/tensors)
     *
     * @throws IllegalStateException if failed to invoke the model
     * @throws IllegalArgumentException if given param is null
     */
    public TensorsData invoke(@NonNull TensorsData inData, @Nullable TensorsInfo inInfo) {
        checkPipelineHandle();

        if (inData == null) {
            throw new IllegalArgumentException("Input tensor data is null");
        }

        TensorsData out = nativeInvoke(mHandle, inData, inInfo);
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
            throw new IllegalArgumentException("The param timeout is invalid");
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
            throw new IllegalArgumentException("Input tensor info is null");
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
