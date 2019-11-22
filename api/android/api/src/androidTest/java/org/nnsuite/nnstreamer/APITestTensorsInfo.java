package org.nnsuite.nnstreamer;

import android.support.test.runner.AndroidJUnit4;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

/**
 * Testcases for TensorsInfo.
 */
@RunWith(AndroidJUnit4.class)
public class APITestTensorsInfo {
    private TensorsInfo mInfo;

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();
        mInfo = new TensorsInfo();
    }

    @After
    public void tearDown() {
        mInfo.close();
    }

    @Test
    public void testAddInfo() {
        try {
            mInfo.addTensorInfo("name1", NNStreamer.TENSOR_TYPE_INT8, new int[]{1});
            assertEquals(1, mInfo.getTensorsCount());

            mInfo.addTensorInfo("name2", NNStreamer.TENSOR_TYPE_UINT8, new int[]{2,2});
            assertEquals(2, mInfo.getTensorsCount());

            mInfo.addTensorInfo(NNStreamer.TENSOR_TYPE_FLOAT32, new int[]{3,3,3});
            assertEquals(3, mInfo.getTensorsCount());
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetInfo() {
        try {
            testAddInfo();

            assertEquals("name1", mInfo.getTensorName(0));
            assertEquals(NNStreamer.TENSOR_TYPE_INT8, mInfo.getTensorType(0));
            assertArrayEquals(new int[]{1,1,1,1}, mInfo.getTensorDimension(0));

            assertEquals("name2", mInfo.getTensorName(1));
            assertEquals(NNStreamer.TENSOR_TYPE_UINT8, mInfo.getTensorType(1));
            assertArrayEquals(new int[]{2,2,1,1}, mInfo.getTensorDimension(1));

            assertNull(mInfo.getTensorName(2));
            assertEquals(NNStreamer.TENSOR_TYPE_FLOAT32, mInfo.getTensorType(2));
            assertArrayEquals(new int[]{3,3,3,1}, mInfo.getTensorDimension(2));

            assertEquals(3, mInfo.getTensorsCount());
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetSize() {
        try {
            testAddInfo();

            /* index 0: 1 int8 */
            assertEquals(1, mInfo.getTensorSize(0));

            /* index 1: 2:2 uint8 */
            assertEquals(4, mInfo.getTensorSize(1));

            /* index 2: 3:3:3 float32 */
            assertEquals(108, mInfo.getTensorSize(2));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAllocate() {
        try {
            testAddInfo();

            TensorsData data = mInfo.allocate();

            assertEquals(3, data.getTensorsCount());
            assertEquals(1, data.getTensorData(0).capacity());
            assertEquals(4, data.getTensorData(1).capacity());
            assertEquals(108, data.getTensorData(2).capacity());
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testUpdateInfo() {
        try {
            testAddInfo();

            mInfo.setTensorName(2, "name3");
            assertEquals("name1", mInfo.getTensorName(0));
            assertEquals("name2", mInfo.getTensorName(1));
            assertEquals("name3", mInfo.getTensorName(2));

            mInfo.setTensorType(2, NNStreamer.TENSOR_TYPE_INT64);
            assertEquals(NNStreamer.TENSOR_TYPE_INT8, mInfo.getTensorType(0));
            assertEquals(NNStreamer.TENSOR_TYPE_UINT8, mInfo.getTensorType(1));
            assertEquals(NNStreamer.TENSOR_TYPE_INT64, mInfo.getTensorType(2));

            mInfo.setTensorDimension(2, new int[]{2,3});
            assertArrayEquals(new int[]{1,1,1,1}, mInfo.getTensorDimension(0));
            assertArrayEquals(new int[]{2,2,1,1}, mInfo.getTensorDimension(1));
            assertArrayEquals(new int[]{2,3,1,1}, mInfo.getTensorDimension(2));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAddInvalidType() {
        try {
            mInfo.addTensorInfo(100, new int[]{2,2,2,2});
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mInfo.getTensorsCount());
    }

    @Test
    public void testAddUnknownType() {
        try {
            mInfo.addTensorInfo(NNStreamer.TENSOR_TYPE_UNKNOWN, new int[]{2,2,2,2});
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mInfo.getTensorsCount());
    }

    @Test
    public void testAddInvalidRank() {
        try {
            mInfo.addTensorInfo(NNStreamer.TENSOR_TYPE_INT32, new int[]{2,2,2,2,2});
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mInfo.getTensorsCount());
    }

    @Test
    public void testAddInvalidDimension() {
        try {
            mInfo.addTensorInfo(NNStreamer.TENSOR_TYPE_INT32, new int[]{1,1,-1});
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mInfo.getTensorsCount());
    }

    @Test
    public void testAddNullDimension() {
        try {
            mInfo.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT8, null);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mInfo.getTensorsCount());
    }

    @Test
    public void testGetInvalidIndex() {
        try {
            mInfo.getTensorType(0);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAddMaxInfo() {
        try {
            for (int i = 0; i <= NNStreamer.TENSOR_SIZE_LIMIT; i++) {
                mInfo.addTensorInfo(NNStreamer.TENSOR_TYPE_FLOAT32, new int[]{2,2,2,2});
            }
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }
}
