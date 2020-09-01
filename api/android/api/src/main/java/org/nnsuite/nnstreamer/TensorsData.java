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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

/**
 * Provides interfaces to handle tensor data frame.
 */
public final class TensorsData implements AutoCloseable {
    private TensorsInfo mInfo = null;
    private ArrayList<ByteBuffer> mDataList = new ArrayList<>();

    /**
     * Allocates a new direct byte buffer with the native byte order.
     *
     * @param size The byte size of the buffer
     *
     * @return The new byte buffer
     *
     * @throws IllegalArgumentException if given size is invalid
     */
    public static ByteBuffer allocateByteBuffer(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Given size is invalid");
        }

        return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
    }

    /**
     * Allocates a new {@link TensorsData} instance with the given tensors information.
     *
     * @param info The tensors information
     *
     * @return {@link TensorsData} instance
     *
     * @throws IllegalArgumentException if given info is invalid
     */
    public static TensorsData allocate(@NonNull TensorsInfo info) {
        if (info == null || info.getTensorsCount() == 0) {
            throw new IllegalArgumentException("Given info is invalid");
        }

        TensorsData data = new TensorsData(info);
        int count = info.getTensorsCount();

        for (int i = 0; i < count; i++) {
            data.addTensorData(allocateByteBuffer(info.getTensorSize(i)));
        }

        return data;
    }

    /**
     * Gets the tensors information.
     *
     * @return {@link TensorsInfo} instance cloned from current tensors information.
     */
    public TensorsInfo getTensorsInfo() {
        return mInfo.clone();
    }

    /**
     * Sets the tensors information.
     *
     * @param info The tensors information
     *
     * @throws IllegalArgumentException if given info is null
     */
    private void setTensorsInfo(@NonNull TensorsInfo info) {
        if (info == null || info.getTensorsCount() == 0) {
            throw new IllegalArgumentException("Given info is invalid");
        }

        mInfo = info.clone();
    }

    /**
     * Gets the number of tensors in tensors data.
     *
     * @return The number of tensors
     */
    public int getTensorsCount() {
        return mDataList.size();
    }

    /**
     * Adds a new tensor data.
     *
     * @param data The tensor data to be added
     *
     * @throws IllegalArgumentException if given data is invalid
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     */
    private void addTensorData(@NonNull ByteBuffer data) {
        int index = getTensorsCount();

        checkByteBuffer(index, data);
        mDataList.add(data);
    }

    /**
     * Internal method called from native to add tensor.
     */
    private void addTensorFromNative(byte[] data) {
        ByteBuffer buffer = allocateByteBuffer(data.length);
        buffer.put(data);

        addTensorData(buffer);
    }

    /**
     * Gets a tensor data of given index.
     *
     * @param index The index of the tensor in the list
     *
     * @return The tensor data
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    public ByteBuffer getTensorData(int index) {
        checkIndexBounds(index);
        return mDataList.get(index);
    }

    /**
     * Sets a tensor data.
     *
     * @param index The index of the tensor in the list
     * @param data  The tensor data
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     * @throws IllegalArgumentException if given data is invalid
     */
    public void setTensorData(int index, @NonNull ByteBuffer data) {
        checkIndexBounds(index);
        checkByteBuffer(index, data);

        mDataList.set(index, data);
    }

    /**
     * Internal method called from native to get the array of tensor data.
     */
    private Object[] getDataArray() {
        return mDataList.toArray();
    }

    /**
     * Internal method to check the index.
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    private void checkIndexBounds(int index) {
        if (index < 0 || index >= getTensorsCount()) {
            throw new IndexOutOfBoundsException("Invalid index [" + index + "] of the tensors");
        }
    }

    /**
     * Internal method to check byte buffer.
     *
     * @throws IllegalArgumentException if given data is invalid
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    private void checkByteBuffer(int index, ByteBuffer data) {
        if (data == null) {
            throw new IllegalArgumentException("Given data is null");
        }

        if (!data.isDirect()) {
            throw new IllegalArgumentException("Given data is not a direct buffer");
        }

        if (data.order() != ByteOrder.nativeOrder()) {
            /* Default byte order of ByteBuffer in java is big-endian, it should be a little-endian. */
            throw new IllegalArgumentException("Given data has invalid byte order");
        }

        if (index >= NNStreamer.TENSOR_SIZE_LIMIT) {
            throw new IndexOutOfBoundsException("Max size of the tensors is " + NNStreamer.TENSOR_SIZE_LIMIT);
        }

        /* compare to tensors info */
        if (mInfo != null) {
            int count = mInfo.getTensorsCount();

            if (index >= count) {
                throw new IndexOutOfBoundsException("Current information has " + count + " tensors");
            }

            int size = mInfo.getTensorSize(index);

            if (data.capacity() != size) {
                throw new IllegalArgumentException("Invalid buffer size, required size is " + size);
            }
        }
    }

    @Override
    public void close() {
        mDataList.clear();
        mInfo = null;
    }

    /**
     * Private constructor to prevent the instantiation.
     */
    private TensorsData(TensorsInfo info) {
        setTensorsInfo(info);
    }
}
