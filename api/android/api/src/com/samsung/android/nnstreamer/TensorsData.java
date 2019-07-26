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

import android.support.annotation.NonNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

/**
 * Provides interfaces to handle tensor data frame.
 */
public final class TensorsData implements AutoCloseable {
    private ArrayList<ByteBuffer> mDataList = new ArrayList<>();

    /**
     * Allocates a new direct byte buffer with the native byte order.
     *
     * @param size The byte size of the buffer
     *
     * @return The new byte buffer
     */
    public static ByteBuffer allocateByteBuffer(int size) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(size);

        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    /**
     * Allocates a new <code>TensorsData</code> instance with the given tensors information.
     *
     * @param info The tensors information
     *
     * @return The allocated tensors data instance
     *
     * @throws IllegalArgumentException if given param is invalid
     */
    public static TensorsData allocate(@NonNull TensorsInfo info) {
        if (info == null) {
            throw new IllegalArgumentException("The param info is null");
        }

        TensorsData data = new TensorsData();
        int count = info.getTensorsCount();

        for (int i = 0; i < count; i++) {
            int type = info.getTesorType(i);
            int[] dimension = info.getTesorDimension(i);

            int size = 0;

            switch (type) {
                case NNStreamer.TENSOR_TYPE_INT32:
                case NNStreamer.TENSOR_TYPE_UINT32:
                case NNStreamer.TENSOR_TYPE_FLOAT32:
                    size = 4;
                    break;
                case NNStreamer.TENSOR_TYPE_INT16:
                case NNStreamer.TENSOR_TYPE_UINT16:
                    size = 2;
                    break;
                case NNStreamer.TENSOR_TYPE_INT8:
                case NNStreamer.TENSOR_TYPE_UINT8:
                    size = 1;
                    break;
                case NNStreamer.TENSOR_TYPE_FLOAT64:
                case NNStreamer.TENSOR_TYPE_INT64:
                case NNStreamer.TENSOR_TYPE_UINT64:
                    size = 8;
                    break;
                default:
                    /* unknown type */
                    break;
            }

            for (int j = 0; j < NNStreamer.TENSOR_RANK_LIMIT; j++) {
                size *= dimension[j];
            }

            data.addTensorData(allocateByteBuffer(size));
        }

        return data;
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
     * @param data The data object to be added
     *
     * @throws IllegalArgumentException if the data is not a byte buffer or the buffer is invalid
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     */
    public void addTensorData(@NonNull Object data) {
        if (data == null || !(data instanceof ByteBuffer)) {
            throw new IllegalArgumentException("Given data is not a byte buffer");
        }

        addTensorData((ByteBuffer) data);
    }

    /**
     * Adds a new tensor data.
     *
     * @param data The byte array to be added
     *
     * @throws IllegalArgumentException if given data is invalid
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     */
    public void addTensorData(@NonNull byte[] data) {
        if (data == null) {
            throw new IllegalArgumentException("Given data is null");
        }

        ByteBuffer buffer = allocateByteBuffer(data.length);
        buffer.put(data);

        addTensorData(buffer);
    }

    /**
     * Adds a new tensor data.
     *
     * @param data The tensor data to be added
     *
     * @throws IllegalArgumentException if given data is invalid
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     */
    public void addTensorData(@NonNull ByteBuffer data) {
        checkByteBuffer(data);

        int index = getTensorsCount();

        if (index >= NNStreamer.TENSOR_SIZE_LIMIT) {
            throw new IndexOutOfBoundsException("Max size of the tensors is " + NNStreamer.TENSOR_SIZE_LIMIT);
        }

        mDataList.add(data);
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
        checkByteBuffer(data);

        mDataList.set(index, data);
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
     */
    private void checkByteBuffer(ByteBuffer data) {
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
    }

    @Override
    public void close() {
        mDataList.clear();
    }
}
