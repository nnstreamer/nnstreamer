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

import java.util.Hashtable;

/**
 * Provides interfaces to create and execute stream pipelines with neural networks.<br>
 * <br>
 * <code>Pipeline</code> allows the following operations with NNStreamer:<br>
 * - Create a stream pipeline with NNStreamer plugins, GStreamer plugins.<br>
 * - Interfaces to push data to the pipeline from the application.<br>
 * - Interfaces to pull data from the pipeline to the application.<br>
 * - Interfaces to start/stop/destroy the pipeline.<br>
 * - Interfaces to control switches and valves in the pipeline.<br>
 */
public final class Pipeline implements AutoCloseable {
    private long mHandle = 0;
    private Hashtable<String, NewDataCallback> mSinkCallbacks = new Hashtable<>();
    private StateChangeCallback mStateCallback = null;

    private native long nativeConstruct(String description, boolean addStateCb);
    private native void nativeDestroy(long handle);
    private native boolean nativeStart(long handle);
    private native boolean nativeStop(long handle);
    private native int nativeGetState(long handle);
    private native boolean nativeInputData(long handle, String name, TensorsData data);
    private native String[] nativeGetSwitchPads(long handle, String name);
    private native boolean nativeSelectSwitchPad(long handle, String name, String pad);
    private native boolean nativeControlValve(long handle, String name, boolean open);
    private native boolean nativeAddSinkCallback(long handle, String name);
    private native boolean nativeRemoveSinkCallback(long handle, String name);

    /**
     * Interface definition for a callback to be invoked when a sink node receives new data.
     */
    public interface NewDataCallback {
        /**
         * Called when a sink node receives new data.
         *
         * If an application wants to accept data outputs of an NNStreamer stream, use this callback to get data from the stream.
         * Note that the buffer may be deallocated after the return and this is synchronously called.
         * Thus, if you need the data afterwards, copy the data to another buffer and return fast.
         * Do not spend too much time in the callback. It is recommended to use very small tensors at sinks.
         *
         * @param data The output data (a single frame, tensor/tensors)
         * @param info The tensors information (dimension, type of output tensor/tensors)
         */
        void onNewDataReceived(TensorsData data, TensorsInfo info);
    }

    /**
     * Interface definition for a callback to be invoked when the pipeline state is changed.
     * This callback can be registered only when constructing the pipeline.<br>
     * <br>
     * The state of pipeline:<br>
     * {@link NNStreamer#PIPELINE_STATE_UNKNOWN}<br>
     * {@link NNStreamer#PIPELINE_STATE_NULL}<br>
     * {@link NNStreamer#PIPELINE_STATE_READY}<br>
     * {@link NNStreamer#PIPELINE_STATE_PAUSED}<br>
     * {@link NNStreamer#PIPELINE_STATE_PLAYING}<br>
     *
     * @see #start()
     * @see #stop()
     */
    public interface StateChangeCallback {
        /**
         * Called when the pipeline state is changed.
         *
         * If an application wants to get the change of pipeline state, use this callback.
         * This callback can be registered when constructing the pipeline.
         * This is synchronously called, so do not spend too much time in the callback.
         *
         * @param state The changed state
         */
        void onStateChanged(int state);
    }

    /**
     * Creates a new <code>Pipeline</code> instance with the given pipeline description.
     *
     * @param description The pipeline description.
     *                    Refer to GStreamer manual or NNStreamer (github.com/nnsuite/nnstreamer) documentation for examples and the grammar.
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public Pipeline(@NonNull String description) {
        this(description, null);
    }

    /**
     * Creates a new <code>Pipeline</code> instance with the given pipeline description.
     *
     * @param description The pipeline description.
     *                    Refer to GStreamer manual or NNStreamer (github.com/nnsuite/nnstreamer) documentation for examples and the grammar.
     * @param callback    The function to be called when the pipeline state is changed.
     *                    You may set null if it's not required.
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public Pipeline(@NonNull String description, @Nullable StateChangeCallback callback) {
        if (description == null) {
            throw new IllegalArgumentException("The param description is null");
        }

        mHandle = nativeConstruct(description, (callback != null));
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to construct the pipeline");
        }

        mStateCallback = callback;
    }

    /**
     * Starts the pipeline, asynchronously.
     * The pipeline state would be changed to 'PLAYING'.
     * If you need to get the changed state, add a callback while constructing a pipeline.
     *
     * @throws IllegalStateException if failed to start the pipeline
     *
     * @see StateChangeCallback
     */
    public void start() {
        checkPipelineHandle();

        if (!nativeStart(mHandle)) {
            throw new IllegalStateException("Failed to start the pipeline");
        }
    }

    /**
     * Stops the pipeline, asynchronously.
     * The pipeline state would be changed to 'PAUSED'.
     * If you need to get the changed state, add a callback while constructing a pipeline.
     *
     * @throws IllegalStateException if failed to stop the pipeline
     *
     * @see StateChangeCallback
     */
    public void stop() {
        checkPipelineHandle();

        if (!nativeStop(mHandle)) {
            throw new IllegalStateException("Failed to stop the pipeline");
        }
    }

    /**
     * Gets the state of pipeline.<br>
     * {@link NNStreamer#PIPELINE_STATE_UNKNOWN}<br>
     * {@link NNStreamer#PIPELINE_STATE_NULL}<br>
     * {@link NNStreamer#PIPELINE_STATE_READY}<br>
     * {@link NNStreamer#PIPELINE_STATE_PAUSED}<br>
     * {@link NNStreamer#PIPELINE_STATE_PLAYING}<br>
     *
     * @return The state of pipeline
     *
     * @throws IllegalStateException if the pipeline is not constructed
     *
     * @see StateChangeCallback
     */
    public int getState() {
        checkPipelineHandle();

        return nativeGetState(mHandle);
    }

    /**
     * Adds an input data frame to source node.
     *
     * @param name The name of source node
     * @param data The input data (a single frame, tensor/tensors)
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to push data to source node
     */
    public void inputData(@NonNull String name, @NonNull TensorsData data) {
        checkPipelineHandle();

        if (name == null) {
            throw new IllegalArgumentException("The param name is null");
        }

        if (data == null) {
            throw new IllegalArgumentException("The param data is null");
        }

        if (!nativeInputData(mHandle, name, data)) {
            throw new IllegalStateException("Failed to push data to source node " + name);
        }
    }

    /**
     * Gets the pad names of a switch.
     *
     * @param name The name of switch node
     *
     * @return The list of pad names
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to get the list of pad names
     */
    public String[] getSwitchPads(@NonNull String name) {
        checkPipelineHandle();

        if (name == null) {
            throw new IllegalArgumentException("The param name is null");
        }

        String[] pads = nativeGetSwitchPads(mHandle, name);

        if (pads == null || pads.length == 0) {
            throw new IllegalStateException("Failed to get the pads in switch " + name);
        }

        return pads;
    }

    /**
     * Controls the switch to select input/output nodes (pads).
     *
     * @param name The name of switch node
     * @param pad  The name of the chosen pad to be activated
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to select the switch pad
     */
    public void selectSwitchPad(@NonNull String name, @NonNull String pad) {
        checkPipelineHandle();

        if (name == null) {
            throw new IllegalArgumentException("The param name is null");
        }

        if (pad == null) {
            throw new IllegalArgumentException("The param pad is null");
        }

        if (!nativeSelectSwitchPad(mHandle, name, pad)) {
            throw new IllegalStateException("Failed to select the pad " + pad);
        }
    }

    /**
     * Controls the valve.
     * Set the flag true to open (let the flow pass), false to close (drop & stop the flow).
     *
     * @param name The name of valve node
     * @param open The flag to control the flow
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to change the valve state
     */
    public void controlValve(@NonNull String name, boolean open) {
        checkPipelineHandle();

        if (name == null) {
            throw new IllegalArgumentException("The param name is null");
        }

        if (!nativeControlValve(mHandle, name, open)) {
            throw new IllegalStateException("Failed to change the valve " + name);
        }
    }

    /**
     * Registers new data callback to sink node.
     * If an application registers a callback with same name, the callback is replaced with new one.
     *
     * @param name     The name of sink node
     * @param callback Callback for new data
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to add the callback to sink node in the pipeline
     */
    public void setSinkCallback(@NonNull String name, NewDataCallback callback) {
        if (name == null) {
            throw new IllegalArgumentException("The param name is null");
        }

        synchronized(this) {
            if (mSinkCallbacks.containsKey(name)) {
                if (callback == null) {
                    /* remove callback */
                    mSinkCallbacks.remove(name);
                    nativeRemoveSinkCallback(mHandle, name);
                } else {
                    mSinkCallbacks.replace(name, callback);
                }
            } else {
                if (callback == null) {
                    throw new IllegalArgumentException("The param callback is null");
                } else {
                    if (nativeAddSinkCallback(mHandle, name)) {
                        mSinkCallbacks.put(name, callback);
                    } else {
                        throw new IllegalStateException("Failed to set sink callback to " + name);
                    }
                }
            }
        }
    }

    /**
     * Internal method called from native when a new data is available.
     */
    private void newDataReceived(String name, TensorsData data, TensorsInfo info) {
        NewDataCallback cb;

        synchronized(this) {
            cb = mSinkCallbacks.get(name);
        }

        if (cb != null) {
            cb.onNewDataReceived(data, info);
        }
    }

    /**
     * Internal method called from native when the state of pipeline is changed.
     */
    private void stateChanged(int state) {
        StateChangeCallback cb;

        synchronized(this) {
            cb = mStateCallback;
        }

        if (cb != null) {
            cb.onStateChanged(state);
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
        synchronized(this) {
            mSinkCallbacks.clear();
            mStateCallback = null;
        }

        if (mHandle != 0) {
            nativeDestroy(mHandle);
            mHandle = 0;
        }
    }
}
