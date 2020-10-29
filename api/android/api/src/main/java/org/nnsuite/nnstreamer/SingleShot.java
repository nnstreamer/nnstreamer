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

    private native long nativeOpen(String[] models, TensorsInfo inputInfo, TensorsInfo outputInfo, int fw, String custom);
    private native void nativeClose(long handle);
    private native TensorsData nativeInvoke(long handle, TensorsData inputData);
    private native TensorsInfo nativeGetInputInfo(long handle);
    private native TensorsInfo nativeGetOutputInfo(long handle);
    private native boolean nativeSetProperty(long handle, String name, String value);
    private native String nativeGetProperty(long handle, String name);
    private native boolean nativeSetInputInfo(long handle, TensorsInfo inputInfo);
    private native boolean nativeSetTimeout(long handle, int timeout);

    /**
     * Creates a new {@link SingleShot} instance with the given model for TensorFlow Lite.
     * If the model has flexible data dimensions, the pipeline will not be constructed and this will make an exception.
     *
     * @param model The path to the neural network model file
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if this failed to construct the pipeline
     */
    public SingleShot(@NonNull File model) {
        this(new File[]{model}, null, null, NNStreamer.NNFWType.TENSORFLOW_LITE, null);
    }

    /**
     * Creates a new {@link SingleShot} instance with the given model.
     * If the model has flexible data dimensions, the pipeline will not be constructed and this will make an exception.
     *
     * @param model The {@link File} object to the neural network model file
     * @param fw    The neural network framework
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if this failed to construct the pipeline
     *
     * @see NNStreamer#isAvailable(NNStreamer.NNFWType)
     */
    public SingleShot(@NonNull File model, NNStreamer.NNFWType fw) {
        this(new File[]{model}, null, null, fw, null);
    }

    /**
     * Creates a new {@link SingleShot} instance with the given model and custom option.
     * If the model has flexible data dimensions, the pipeline will not be constructed and this will make an exception.
     *
     * @param model  The {@link File} object to the neural network model file
     * @param fw     The neural network framework
     * @param custom The custom option string to open the neural network
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if this failed to construct the pipeline
     *
     * @see NNStreamer#isAvailable(NNStreamer.NNFWType)
     */
    public SingleShot(@NonNull File model, NNStreamer.NNFWType fw, @Nullable String custom) {
        this(new File[]{model}, null, null, fw, custom);
    }

    /**
     * Creates a new {@link SingleShot} instance with the given model for TensorFlow Lite.
     * The input and output tensors information are required if the given model has flexible data dimensions,
     * where the information MUST be given before executing the model.
     * However, once it's given, the dimension cannot be changed for the given model handle.
     * You may set null if it's not required.
     *
     * @param model         The {@link File} object to the neural network model file
     * @param inputInfo     The input tensors information
     * @param outputInfo    The output tensors information
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException    if this failed to construct the pipeline
     */
    public SingleShot(@NonNull File model, @Nullable TensorsInfo inputInfo, @Nullable TensorsInfo outputInfo) {
        this(new File[]{model}, inputInfo, outputInfo, NNStreamer.NNFWType.TENSORFLOW_LITE, null);
    }

    /**
     * Creates a new {@link SingleShot} instance with the given files and custom option.
     *
     * Unlike other constructors, this handles multiple files and custom option string
     * when the neural network requires various options and model files.
     *
     * @param models        The array of {@link File} objects to the neural network model files
     * @param inputInfo     The input tensors information
     * @param outputInfo    The output tensors information
     * @param fw            The neural network framework
     * @param custom        The custom option string to open the neural network
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException    if this failed to construct the pipeline
     *
     * @see NNStreamer#isAvailable(NNStreamer.NNFWType)
     */
    public SingleShot(@NonNull File[] models, @Nullable TensorsInfo inputInfo, @Nullable TensorsInfo outputInfo,
                      NNStreamer.NNFWType fw, @Nullable String custom) {
        this(new Options(fw, models, inputInfo, outputInfo, custom));
    }

    /**
     * Creates a new {@link SingleShot} instance with the given {@link Options}.
     *
     * @param options   The {@link Options} object configuring the instance
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException    if this failed to construct the pipeline
     */
    public SingleShot(@NonNull Options options) {
        File[] models = options.getModels();
        NNStreamer.NNFWType fw = options.getNNFWType();
        TensorsInfo inputInfo = options.getInputInfo();
        TensorsInfo outputInfo = options.getOutputInfo();
        String custom = options.getCustom();

        String[] path = new String[models.length];
        int index = 0;

        for (File model : models) {
            path[index++] = model.getAbsolutePath();
        }

        mHandle = nativeOpen(path, inputInfo, outputInfo, fw.ordinal(), custom);
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to construct the SingleShot instance");
        }
    }

    /**
     * Invokes the model with the given input data.
     *
     * Even if the model has flexible input data dimensions,
     * input data frames of an instance of a model should share the same dimension.
     * To change the input information, you should call {@link #setInputInfo(TensorsInfo)} before calling invoke method.
     *
     * Note that this will wait for the result until the invoke process is done.
     * If an application wants to change the time to wait for an output,
     * set the timeout using {@link #setTimeout(int)}.
     *
     * @param in The input data to be inferred (a single frame, tensor/tensors)
     *
     * @return The output data (a single frame, tensor/tensors)
     *
     * @throws IllegalStateException if this failed to invoke the model
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
     * @throws IllegalStateException if this failed to get the input information
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
     * @throws IllegalStateException if this failed to get the output information
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
     * @throws IllegalStateException if this failed to set the timeout
     */
    public void setTimeout(int timeout) {
        checkPipelineHandle();

        if (timeout < 0) {
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
     * Note that a model/framework may not support changing the information.
     *
     * @param in The input tensors information
     *
     * @throws IllegalStateException if this failed to set the input information
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

    /**
     * Provides interfaces to configure SingleShot instance.
     */
    public static class Options {
        private NNStreamer.NNFWType fw = NNStreamer.NNFWType.UNKNOWN;
        private File[] models;
        private TensorsInfo inputInfo;
        private TensorsInfo outputInfo;
        private String custom;

        /**
         * Creates a new {@link Options} instance with the given framework and file.
         *
         * @param type  The type of {@link NNStreamer.NNFWType}
         * @param model The {@link File} object to the neural network model file
         *
         * @throws IllegalArgumentException if given model is invalid
         * @throws IllegalStateException    if given framework is not available
         */
        public Options(NNStreamer.NNFWType type, File model) {
            setNNFWType(type);
            setModels(new File[]{model});
        }

        /**
         * Creates a new {@link Options} instance with the given framework and file.
         *
         * @param type   The type of {@link NNStreamer.NNFWType}
         * @param models The array of {@link File} objects to the neural network model files
         *
         * @throws IllegalArgumentException if given models is invalid
         * @throws IllegalStateException    if given framework is not available
         */
        public Options(NNStreamer.NNFWType type, File[] models) {
            setNNFWType(type);
            setModels(models);
        }

        /**
         * Creates a new {@link Options} instance with the given parameters.
         *
         * @param type              The type of {@link NNStreamer.NNFWType}
         * @param models            The array of {@link File} objects to the neural network model files
         * @param inputInfo         The input tensors information
         * @param outputInfo        The output tensors information
         * @param custom            The custom option string to open the neural network instance
         *
         * @throws IllegalArgumentException if given models is invalid
         * @throws IllegalStateException    if given framework is not available
         */
        public Options(NNStreamer.NNFWType type, File[] models, TensorsInfo inputInfo, TensorsInfo outputInfo, String custom) {
            setNNFWType(type);
            setModels(models);
            setInputInfo(inputInfo);
            setOutputInfo(outputInfo);
            setCustom(custom);
        }

        public NNStreamer.NNFWType getNNFWType() {
            return fw;
        }

        public void setNNFWType(NNStreamer.NNFWType fw) {
            if (!NNStreamer.isAvailable(fw)) {
                throw new IllegalStateException("Given framework " + fw.name() + " is not available");
            }
            this.fw = fw;
        }

        public File[] getModels() {
            return models;
        }

        public void setModels(File[] models) {
            if (models == null) {
                throw new IllegalArgumentException("Given model is invalid");
            }

            for (File model : models) {
                if (model == null || !model.exists()) {
                    throw new IllegalArgumentException("Given model is invalid");
                }
            }

            this.models = models;
        }

        public TensorsInfo getInputInfo() {
            return inputInfo;
        }

        public void setInputInfo(TensorsInfo inputInfo) {
            this.inputInfo = inputInfo;
        }

        public TensorsInfo getOutputInfo() {
            return outputInfo;
        }

        public void setOutputInfo(TensorsInfo outputInfo) {
            this.outputInfo = outputInfo;
        }

        public String getCustom() {
            return custom;
        }

        public void setCustom(String custom) {
            this.custom = custom;
        }
    }
}
