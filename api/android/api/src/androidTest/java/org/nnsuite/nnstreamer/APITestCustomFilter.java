package org.nnsuite.nnstreamer;

import android.support.test.runner.AndroidJUnit4;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Testcases for CustomFilter.
 */
@RunWith(AndroidJUnit4.class)
public class APITestCustomFilter {
    private int mReceived = 0;
    private boolean mInvalidState = false;
    private boolean mRegistered = false;
    private CustomFilter mCustomPassthrough;
    private CustomFilter mCustomConvert;
    private CustomFilter mCustomAdd;

    private Pipeline.NewDataCallback mSinkCb = new Pipeline.NewDataCallback() {
        @Override
        public void onNewDataReceived(TensorsData data) {
            if (data == null || data.getTensorsCount() != 1) {
                mInvalidState = true;
                return;
            }

            TensorsInfo info = data.getTensorsInfo();

            if (info == null || info.getTensorsCount() != 1) {
                mInvalidState = true;
                return;
            }

            ByteBuffer output = data.getTensorData(0);

            for (int i = 0; i < 10; i++) {
                float expected = i + 1.5f;

                if (expected != output.getFloat(i * 4)) {
                    mInvalidState = true;
                }
            }

            mReceived++;
        }
    };

    private void registerCustomFilters() {
        try {
            TensorsInfo inputInfo = new TensorsInfo();
            inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

            TensorsInfo outputInfo = inputInfo.clone();

            /* register custom-filter (passthrough) */
            mCustomPassthrough = CustomFilter.registerCustomFilter("custom-passthrough",
                    inputInfo, outputInfo, new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    return in;
                }
            });

            /* register custom-filter (convert data type to float) */
            outputInfo.setTensorType(0, NNStreamer.TensorType.FLOAT32);
            mCustomConvert = CustomFilter.registerCustomFilter("custom-convert",
                    inputInfo, outputInfo, new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    TensorsInfo info = in.getTensorsInfo();
                    ByteBuffer input = in.getTensorData(0);

                    info.setTensorType(0, NNStreamer.TensorType.FLOAT32);

                    TensorsData out = info.allocate();
                    ByteBuffer output = out.getTensorData(0);

                    for (int i = 0; i < 10; i++) {
                        float value = (float) input.getInt(i * 4);
                        output.putFloat(i * 4, value);
                    }

                    out.setTensorData(0, output);
                    return out;
                }
            });

            /* register custom-filter (add constant) */
            inputInfo.setTensorType(0, NNStreamer.TensorType.FLOAT32);
            mCustomAdd = CustomFilter.registerCustomFilter("custom-add",
                    inputInfo, outputInfo, new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    TensorsInfo info = in.getTensorsInfo();
                    ByteBuffer input = in.getTensorData(0);

                    TensorsData out = info.allocate();
                    ByteBuffer output = out.getTensorData(0);

                    for (int i = 0; i < 10; i++) {
                        float value = input.getFloat(i * 4);

                        /* add constant */
                        value += 1.5f;
                        output.putFloat(i * 4, value);
                    }

                    out.setTensorData(0, output);
                    return out;
                }
            });

            mRegistered = true;
        } catch (Exception e) {
            /* failed to register custom-filters */
            fail();
        }
    }

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        mReceived = 0;
        mInvalidState = false;

        if (!mRegistered) {
            registerCustomFilters();
        }
    }

    @After
    public void tearDown() {
        if (mRegistered) {
            mCustomPassthrough.close();
            mCustomConvert.close();
            mCustomAdd.close();
        }
    }

    @Test
    public void testGetName() {
        try {
            assertEquals("custom-passthrough", mCustomPassthrough.getName());
            assertEquals("custom-convert", mCustomConvert.getName());
            assertEquals("custom-add", mCustomAdd.getName());
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testCustomFilters() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)10:1:1:1,type=(string)int32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=custom-easy model=" + mCustomPassthrough.getName() + " ! " +
                "tensor_filter framework=custom-easy model=" + mCustomConvert.getName() + " ! " +
                "tensor_filter framework=custom-easy model=" + mCustomAdd.getName() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer repeatedly */
            for (int i = 0; i < 2048; i++) {
                TensorsData in = TensorsData.allocate(info);
                ByteBuffer input = in.getTensorData(0);

                for (int j = 0; j < 10; j++) {
                    input.putInt(j * 4, j);
                }

                in.setTensorData(0, input);

                pipe.inputData("srcx", in);
                Thread.sleep(20);
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(2048, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testDropBuffer() {
        TensorsInfo inputInfo = new TensorsInfo();
        inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10,1,1,1});

        TensorsInfo outputInfo = inputInfo.clone();

        CustomFilter customDrop = CustomFilter.registerCustomFilter("custom-drop",
                inputInfo, outputInfo, new CustomFilter.CustomFilterCallback() {
            int received = 0;

            @Override
            public TensorsData invoke(TensorsData in) {
                received++;

                if (received <= 5) {
                    return in;
                }

                /* return null to drop the incoming buffer */
                return null;
            }
        });

        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)10:1:1:1,type=(string)int32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=custom-easy model=" + customDrop.getName() + " ! " +
                "tensor_filter framework=custom-easy model=" + mCustomPassthrough.getName() + " ! " +
                "tensor_filter framework=custom-easy model=" + mCustomConvert.getName() + " ! " +
                "tensor_filter framework=custom-easy model=" + mCustomAdd.getName() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10,1,1,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer repeatedly */
            for (int i = 0; i < 24; i++) {
                TensorsData in = TensorsData.allocate(info);
                ByteBuffer input = in.getTensorData(0);

                for (int j = 0; j < 10; j++) {
                    input.putInt(j * 4, j);
                }

                in.setTensorData(0, input);

                pipe.inputData("srcx", in);
                Thread.sleep(20);
            }

            /* sleep 300 to pass input buffers to sink */
            Thread.sleep(300);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(5, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRegisterNullName_n() {
        TensorsInfo inputInfo = new TensorsInfo();
        inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

        TensorsInfo outputInfo = inputInfo.clone();

        try {
            CustomFilter.registerCustomFilter(null, inputInfo, outputInfo,
                    new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    return in;
                }
            });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterNullInputInfo_n() {
        TensorsInfo outputInfo = new TensorsInfo();
        outputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

        try {
            CustomFilter.registerCustomFilter("custom-invalid-info", null, outputInfo,
                    new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    return in;
                }
            });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterNullOutputInfo_n() {
        TensorsInfo inputInfo = new TensorsInfo();
        inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

        try {
            CustomFilter.registerCustomFilter("custom-invalid-info", inputInfo, null,
                    new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    return in;
                }
            });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterNullCallback_n() {
        TensorsInfo inputInfo = new TensorsInfo();
        inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

        TensorsInfo outputInfo = inputInfo.clone();

        try {
            CustomFilter.registerCustomFilter("custom-invalid-cb", inputInfo, outputInfo, null);

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDuplicatedName_n() {
        TensorsInfo inputInfo = new TensorsInfo();
        inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

        TensorsInfo outputInfo = inputInfo.clone();

        try {
            CustomFilter.registerCustomFilter(mCustomPassthrough.getName(), inputInfo, outputInfo,
                    new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    return in;
                }
            });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterPreservedName_n() {
        TensorsInfo inputInfo = new TensorsInfo();
        inputInfo.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

        TensorsInfo outputInfo = inputInfo.clone();

        try {
            CustomFilter.registerCustomFilter("auto", inputInfo, outputInfo,
                    new CustomFilter.CustomFilterCallback() {
                @Override
                public TensorsData invoke(TensorsData in) {
                    return in;
                }
            });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }
}
