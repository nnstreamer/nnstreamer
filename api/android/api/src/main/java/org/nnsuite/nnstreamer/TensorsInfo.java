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

import java.util.ArrayList;

/**
 * Provides interfaces to handle tensors information.<br>
 * <br>
 * The data type of tensor in NNStreamer:<br>
 * {@link NNStreamer#TENSOR_TYPE_INT32}<br>
 * {@link NNStreamer#TENSOR_TYPE_UINT32}<br>
 * {@link NNStreamer#TENSOR_TYPE_INT16}<br>
 * {@link NNStreamer#TENSOR_TYPE_UINT16}<br>
 * {@link NNStreamer#TENSOR_TYPE_INT8}<br>
 * {@link NNStreamer#TENSOR_TYPE_UINT8}<br>
 * {@link NNStreamer#TENSOR_TYPE_FLOAT64}<br>
 * {@link NNStreamer#TENSOR_TYPE_FLOAT32}<br>
 *
 * @see NNStreamer#TENSOR_RANK_LIMIT
 * @see NNStreamer#TENSOR_SIZE_LIMIT
 */
public final class TensorsInfo implements AutoCloseable {
    private ArrayList<TensorInfo> mInfoList = new ArrayList<>();

    /**
     * Allocates a new {@link TensorsData} instance with the tensors information.
     *
     * @return {@link TensorsData} instance
     *
     * @throws IllegalStateException if tensors info is empty
     */
    public TensorsData allocate() {
        if (getTensorsCount() == 0) {
            throw new IllegalStateException("Empty tensor info");
        }

        return TensorsData.allocate(this);
    }

    /**
     * Gets the number of tensors.
     * The maximum number of tensors is {@link NNStreamer#TENSOR_SIZE_LIMIT}.
     *
     * @return The number of tensors
     */
    public int getTensorsCount() {
        return mInfoList.size();
    }

    /**
     * Adds a new tensor information.
     *
     * @param type      The tensor data type
     * @param dimension The tensor dimension
     *
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     * @throws IllegalArgumentException if given param is null or invalid
     */
    public void addTensorInfo(int type, @NonNull int[] dimension) {
        addTensorInfo(null, type, dimension);
    }

    /**
     * Adds a new tensor information.
     *
     * @param name      The tensor name
     * @param type      The tensor data type
     * @param dimension The tensor dimension
     *
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     * @throws IllegalArgumentException if given param is null or invalid
     */
    public void addTensorInfo(@Nullable String name, int type, @NonNull int[] dimension) {
        int index = getTensorsCount();

        if (index >= NNStreamer.TENSOR_SIZE_LIMIT) {
            throw new IndexOutOfBoundsException("Max number of the tensors is " + NNStreamer.TENSOR_SIZE_LIMIT);
        }

        mInfoList.add(new TensorInfo(name, type, dimension));
    }

    /**
     * Sets the tensor name.
     *
     * @param index The index of the tensor information in the list
     * @param name  The tensor name
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    public void setTensorName(int index, String name) {
        checkIndexBounds(index);
        mInfoList.get(index).setName(name);
    }

    /**
     * Gets the tensor name of given index.
     *
     * @param index The index of the tensor information in the list
     *
     * @return The tensor name
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    public String getTensorName(int index) {
        checkIndexBounds(index);
        return mInfoList.get(index).getName();
    }

    /**
     * Sets the tensor data type.
     *
     * @param index The index of the tensor information in the list
     * @param type  The tensor type
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     * @throws IllegalArgumentException if the given type is unknown or unsupported type
     */
    public void setTensorType(int index, int type) {
        checkIndexBounds(index);
        mInfoList.get(index).setType(type);
    }

    /**
     * Gets the tensor data type of given index.
     *
     * @param index The index of the tensor information in the list
     *
     * @return The tensor data type
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    public int getTensorType(int index) {
        checkIndexBounds(index);
        return mInfoList.get(index).getType();
    }

    /**
     * Sets the tensor dimension
     *
     * @param index     The index of the tensor information in the list
     * @param dimension The tensor dimension
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     * @throws IllegalArgumentException if the given dimension is null or invalid
     */
    public void setTensorDimension(int index, @NonNull int[] dimension) {
        checkIndexBounds(index);
        mInfoList.get(index).setDimension(dimension);
    }

    /**
     * Gets the tensor dimension of given index.
     *
     * @param index The index of the tensor information in the list
     *
     * @return The tensor dimension
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     */
    public int[] getTensorDimension(int index) {
        checkIndexBounds(index);
        return mInfoList.get(index).getDimension();
    }

    /**
     * Calculates the byte size of tensor data.
     *
     * @param index The index of the tensor information in the list
     *
     * @return The byte size of tensor
     *
     * @throws IndexOutOfBoundsException if the given index is invalid
     * @throws IllegalStateException if data type or dimension is invalid
     */
    public int getTensorSize(int index) {
        checkIndexBounds(index);

        int size = mInfoList.get(index).getSize();
        if (size <= 0) {
            throw new IllegalStateException("Unknown data type or invalid dimension");
        }

        return size;
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

    @Override
    public void close() {
        mInfoList.clear();
    }

    /**
     * Internal class for tensor information.
     */
    private static class TensorInfo {
        private String name = null;
        private int type = NNStreamer.TENSOR_TYPE_UNKNOWN;
        private int[] dimension = new int[NNStreamer.TENSOR_RANK_LIMIT];

        public TensorInfo(@Nullable String name, int type, @NonNull int[] dimension) {
            setName(name);
            setType(type);
            setDimension(dimension);
        }

        public void setName(@Nullable String name) {
            this.name = name;
        }

        public String getName() {
            return this.name;
        }

        public void setType(int type) {
            if (type < 0 || type >= NNStreamer.TENSOR_TYPE_UNKNOWN) {
                throw new IllegalArgumentException("Given tensor type is unknown or unsupported type");
            }

            this.type = type;
        }

        public int getType() {
            return this.type;
        }

        public void setDimension(@NonNull int[] dimension) {
            if (dimension == null) {
                throw new IllegalArgumentException("Given tensor dimension is null");
            }

            int rank = dimension.length;

            if (rank > NNStreamer.TENSOR_RANK_LIMIT) {
                throw new IllegalArgumentException("Max size of the tensor rank is " + NNStreamer.TENSOR_RANK_LIMIT);
            }

            for (int dim : dimension) {
                if (dim <= 0) {
                    throw new IllegalArgumentException("The dimension should be a positive value");
                }
            }

            System.arraycopy(dimension, 0, this.dimension, 0, rank);

            /* fill default value */
            for (int i = rank; i < NNStreamer.TENSOR_RANK_LIMIT; i++) {
                this.dimension[i] = 1;
            }
        }

        public int[] getDimension() {
            return this.dimension;
        }

        public int getSize() {
            int size = 0;

            switch (this.type) {
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

            for (int i = 0; i < NNStreamer.TENSOR_RANK_LIMIT; i++) {
                size *= this.dimension[i];
            }

            return size;
        }
    }
}
