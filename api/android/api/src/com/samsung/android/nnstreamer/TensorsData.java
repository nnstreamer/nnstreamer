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
     * @throws IllegalArgumentException if the data is not a byte buffer
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
     * @param data The tensor data to be added
     *
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     */
    public void addTensorData(@NonNull ByteBuffer data) {
        int index = getTensorsCount();

        if (index >= NNStreamer.TENSOR_SIZE_LIMIT) {
            throw new IndexOutOfBoundsException("Max size of the tensors is " + NNStreamer.TENSOR_SIZE_LIMIT);
        }

        mDataList.add(convertBuffer(data));
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
     */
    public void setTensorData(int index, @NonNull ByteBuffer data) {
        checkIndexBounds(index);
        mDataList.set(index, convertBuffer(data));
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
     * Internal method to convert the given data to direct buffer.
     *
     * @param data The tensor data
     *
     * @return The converted buffer
     *
     * @throws IllegalArgumentException if given data is null
     */
    private ByteBuffer convertBuffer(ByteBuffer data) {
        if (data == null) {
            throw new IllegalArgumentException("Given data is null");
        }

        if (data.isDirect() && data.order() == ByteOrder.nativeOrder()) {
            return data;
        }

        ByteBuffer allocated = ByteBuffer.allocateDirect(data.capacity());

        allocated.order(ByteOrder.nativeOrder());
        allocated.put(data);

        return allocated;
    }

    @Override
    public void close() {
        mDataList.clear();
    }
}
