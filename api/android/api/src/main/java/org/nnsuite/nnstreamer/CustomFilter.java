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

/**
 * Provides interfaces to create a custom-filter in the pipeline.<br>
 * <br>
 * To register a new custom-filter, an application should call {@link #registerCustomFilter(String, CustomFilterCallback)}
 * before constructing the pipeline.
 */
public class CustomFilter implements AutoCloseable {
    private long mHandle = 0;
    private String mName = null;
    private CustomFilterCallback mCallback = null;

    private native long nativeInitialize(String name);
    private native void nativeDestroy(long handle);

    /**
     * Interface definition for a callback to be invoked while processing the pipeline.
     */
    public interface CustomFilterCallback {
        /**
         * Called synchronously when constructing a pipeline.
         *
         * NNStreamer filter configures input and output tensors information during the caps negotiation.
         *
         * Note that this is not a fixed value and the pipeline may try different values during the cap negotiations.
         * An application should validate the information of input tensors and return proper output information.
         *
         * @param inInfo The input tensors information
         *
         * @return The output tensors information
         */
        TensorsInfo getOutputInfo(TensorsInfo inInfo);

        /**
         * Called synchronously while processing the pipeline.
         *
         * NNStreamer filter invokes the given custom-filter callback while processing the pipeline.
         *
         * @param inData  The input data (a single frame, tensor/tensors)
         * @param inInfo  The input tensors information
         * @param outInfo The output tensors information
         *
         * @return The output data (a single frame, tensor/tensors)
         */
        TensorsData invoke(TensorsData inData, TensorsInfo inInfo, TensorsInfo outInfo);
    }

    /**
     * Registers new custom-filter with name.
     *
     * Note that if given name is duplicated in the pipeline, the registration will be failed and throw an exception.
     *
     * @param name     The name of custom-filter
     * @param callback The function to be called while processing the pipeline
     *
     * @return <code>CustomFilter</code> instance
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to initialize custom-filter
     */
    public static CustomFilter registerCustomFilter(@NonNull String name, @NonNull CustomFilterCallback callback) {
        return new CustomFilter(name, callback);
    }

    /**
     * Gets the name of custom-filter.
     *
     * @return The name of custom-filter
     */
    public String getName() {
        return mName;
    }

    /**
     * Internal constructor to create and register a custom-filter.
     *
     * @param name     The name of custom-filter
     * @param callback The function to be called while processing the pipeline
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to initialize custom-filter
     */
    private CustomFilter(@NonNull String name, @NonNull CustomFilterCallback callback) {
        if (name == null) {
            throw new IllegalArgumentException("The param name is null");
        }

        if (callback == null) {
            throw new IllegalArgumentException("The param callback is null");
        }

        mName = name;
        mCallback = callback;

        mHandle = nativeInitialize(name);
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to initialize custom-filter " + name);
        }
    }

    /**
     * Internal method called from native during the caps negotiation.
     */
    private TensorsInfo getOutputInfo(TensorsInfo info) {
        TensorsInfo out = null;

        if (mCallback != null) {
            out = mCallback.getOutputInfo(info);
        }

        return out;
    }

    /**
     * Internal method called from native while processing the pipeline.
     */
    private TensorsData invoke(TensorsData inData, TensorsInfo inInfo, TensorsInfo outInfo) {
        TensorsData out = null;

        if (mCallback != null) {
            out = mCallback.invoke(inData, inInfo, outInfo);
        }

        return out;
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
            nativeDestroy(mHandle);
            mHandle = 0;
        }
    }
}
