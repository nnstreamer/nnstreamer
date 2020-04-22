package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;
import android.os.Build;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Testcases for Pipeline.
 */
@RunWith(AndroidJUnit4.class)
public class APITestPipeline {
    private int mReceived = 0;
    private boolean mInvalidState = false;
    private Pipeline.State mPipelineState = Pipeline.State.NULL;

    private Pipeline.NewDataCallback mSinkCb = new Pipeline.NewDataCallback() {
        @Override
        public void onNewDataReceived(TensorsData data) {
            if (data == null ||
                data.getTensorsCount() != 1 ||
                data.getTensorData(0).capacity() != 200) {
                mInvalidState = true;
                return;
            }

            TensorsInfo info = data.getTensorsInfo();

            /* validate received data (unit8 2:10:10:1) */
            if (info == null ||
                info.getTensorsCount() != 1 ||
                info.getTensorName(0) != null ||
                info.getTensorType(0) != NNStreamer.TensorType.UINT8 ||
                !Arrays.equals(info.getTensorDimension(0), new int[]{2,10,10,1})) {
                /* received data is invalid */
                mInvalidState = true;
            }

            mReceived++;
        }
    };

    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        mReceived = 0;
        mInvalidState = false;
        mPipelineState = Pipeline.State.NULL;
    }

    @Test
    public void enumPipelineState() {
        assertEquals(Pipeline.State.UNKNOWN, Pipeline.State.valueOf("UNKNOWN"));
        assertEquals(Pipeline.State.NULL, Pipeline.State.valueOf("NULL"));
        assertEquals(Pipeline.State.READY, Pipeline.State.valueOf("READY"));
        assertEquals(Pipeline.State.PAUSED, Pipeline.State.valueOf("PAUSED"));
        assertEquals(Pipeline.State.PLAYING, Pipeline.State.valueOf("PLAYING"));
    }

    @Test
    public void testConstructInvalidElement_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "invalidelement ! tensor_converter ! tensor_sink";

        try {
            new Pipeline(desc);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructNullDescription_n() {
        try {
            new Pipeline(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructEmptyDescription_n() {
        try {
            new Pipeline("");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructNullStateCb() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        try (Pipeline pipe = new Pipeline(desc, null)) {
            Thread.sleep(100);
            assertEquals(Pipeline.State.PAUSED, pipe.getState());
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testConstructWithStateCb() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        /* pipeline state callback */
        Pipeline.StateChangeCallback stateCb = new Pipeline.StateChangeCallback() {
            @Override
            public void onStateChanged(Pipeline.State state) {
                mPipelineState = state;
            }
        };

        try (Pipeline pipe = new Pipeline(desc, stateCb)) {
            Thread.sleep(100);
            assertEquals(Pipeline.State.PAUSED, mPipelineState);

            /* start pipeline */
            pipe.start();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PLAYING, mPipelineState);

            /* stop pipeline */
            pipe.stop();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PAUSED, mPipelineState);
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetState() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PLAYING, pipe.getState());

            /* stop pipeline */
            pipe.stop();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PAUSED, pipe.getState());
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRegisterNullDataCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("sinkx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("invalid_sink", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback(null, mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterNullDataCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("sinkx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterDataCbNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback(null, mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterDataCbEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisteredDataCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterInvalidCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register callback */
            Pipeline.NewDataCallback cb1 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            pipe.registerSinkCallback("sinkx", cb1);

            /* unregistered callback */
            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRemoveDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testDuplicatedDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* try to register same cb */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testMultipleDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register three callbacks */
            Pipeline.NewDataCallback cb1 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            Pipeline.NewDataCallback cb2 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            pipe.registerSinkCallback("sinkx", mSinkCb);
            pipe.registerSinkCallback("sinkx", cb1);
            pipe.registerSinkCallback("sinkx", cb2);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            pipe.unregisterSinkCallback("sinkx", cb1);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(40, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRunModel() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        File model = APITestCommon.getTFLiteImgModel();
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=tensorflow-lite model=" + model.getAbsolutePath() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3,224,224,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 1001)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(100);
            }

            /* sleep 500 to invoke */
            Thread.sleep(500);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testClassificationResult() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        File model = APITestCommon.getTFLiteImgModel();
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=tensorflow-lite model=" + model.getAbsolutePath() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3,224,224,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    ByteBuffer buffer = data.getTensorData(0);
                    int labelIndex = APITestCommon.getMaxScore(buffer);

                    /* check label index (orange) */
                    if (labelIndex != 951) {
                        mInvalidState = true;
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            TensorsData in = APITestCommon.readRawImageData();
            pipe.inputData("srcx", in);

            /* sleep 1000 to invoke */
            Thread.sleep(1000);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputBuffer() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer repeatedly */
            for (int i = 0; i < 2048; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(20);
            }

            /* sleep 300 to pass input buffers to sink */
            Thread.sleep(300);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData("invalid_src", TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData(null, TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData("", TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputNullData_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            pipe.inputData("srcx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputInvalidData_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{4,10,10,2});

            TensorsData in = TensorsData.allocate(info);

            /* push data with invalid size */
            pipe.inputData("srcx", in);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectSwitch() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(50);

                if (i == 9) {
                    /* select pad */
                    pipe.selectSwitchPad("outs", "src_1");
                }
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetSwitchPad() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list of output-selector */
            String[] pads = pipe.getSwitchPads("outs");

            assertEquals(2, pads.length);
            assertEquals("src_0", pads[0]);
            assertEquals("src_1", pads[1]);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetSwitchInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list with invalid switch name */
            pipe.getSwitchPads("invalid_outs");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetSwitchNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list with null param */
            pipe.getSwitchPads(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetSwitchEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list with empty name */
            pipe.getSwitchPads("");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectInvalidPad_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* select invalid pad name */
            pipe.selectSwitchPad("outs", "invalid_src");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectNullPad_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* null pad name */
            pipe.selectSwitchPad("outs", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectEmptyPad_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* empty pad name */
            pipe.selectSwitchPad("outs", "");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectNullSwitchName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* null switch name */
            pipe.selectSwitchPad(null, "src_1");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectEmptySwitchName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* empty switch name */
            pipe.selectSwitchPad("", "src_1");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testControlValve() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);

                if (i == 9) {
                    /* close valve */
                    pipe.controlValve("valvex", false);
                }
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testControlInvalidValve_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* control valve with invalid name */
            pipe.controlValve("invalid_valve", false);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testControlNullValveName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* control valve with null name */
            pipe.controlValve(null, false);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testControlEmptyValveName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* control valve with empty name */
            pipe.controlValve("", false);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAMCsrc() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        String media = root + "/nnstreamer/test/test_video.mp4";

        String desc = "amcsrc location=" + media + " ! " +
                "videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240 ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 230400)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* sleep 2 seconds to invoke */
            Thread.sleep(2000);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);

            /* sleep 1 second and restart */
            Thread.sleep(1000);
            mReceived = 0;

            pipe.start();

            /* sleep 2 seconds to invoke */
            Thread.sleep(2000);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Run SNAP with Caffe model.
     */
    private void runSNAPCaffe(APITestCommon.SNAPComputingUnit CUnit) {
        File[] models = APITestCommon.getSNAPCaffeModel();
        String option = APITestCommon.getSNAPCaffeOption(CUnit);

        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)float32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=snap " +
                    "model=" + models[0].getAbsolutePath() + "," + models[1].getAbsolutePath() + " " +
                    "input=3:224:224:1 inputtype=float32 inputlayout=NHWC inputname=data " +
                    "output=1:1:1000:1 outputtype=float32 outputlayout=NCHW outputname=prob " +
                    "custom=" + option + " ! " +
                "tensor_sink name=sinkx";

        try (
            Pipeline pipe = new Pipeline(desc);
            TensorsInfo info = new TensorsInfo()
        ) {
            info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{3,224,224,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 4000)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(100);
            }

            /* sleep 500 to invoke */
            Thread.sleep(500);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSNAPCaffeCPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPCaffe(APITestCommon.SNAPComputingUnit.CPU);
    }

    @Test
    public void testSNAPCaffeGPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPCaffe(APITestCommon.SNAPComputingUnit.GPU);
    }

    /**
     * Run SNAP with Tensorflow model.
     */
    private void runSNAPTensorflow(APITestCommon.SNAPComputingUnit CUnit) {
        File[] model = APITestCommon.getSNAPTensorflowModel(CUnit);
        String option = APITestCommon.getSNAPTensorflowOption(CUnit);
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)float32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=snap " +
                    "model=" + model[0].getAbsolutePath() + " " +
                    "input=3:224:224:1 inputtype=float32 inputlayout=NHWC inputname=input " +
                    "output=1001:1 outputtype=float32 outputlayout=NHWC outputname=MobilenetV1/Predictions/Reshape_1:0 " +
                    "custom=" + option + " ! " +
                "tensor_sink name=sinkx";

        try (
            Pipeline pipe = new Pipeline(desc);
            TensorsInfo info = new TensorsInfo()
        ) {
            info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{3,224,224,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 4004)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(100);
            }

            /* sleep 500 to invoke */
            Thread.sleep(500);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSNAPTensorflowCPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPTensorflow(APITestCommon.SNAPComputingUnit.CPU);
    }

    @Test
    public void testSNAPTensorflowDSP() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        if (!android.os.Build.HARDWARE.equals("qcom")) {
            /** 
             * Tensorflow model using DSP runtime can only be executed on 
             * Snapdragon SoC. Cannot run this test on exynos.
             */
            return;
        }

        runSNAPTensorflow(APITestCommon.SNAPComputingUnit.DSP);
    }

    @Test
    public void testSNAPTensorflowNPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        if (!android.os.Build.HARDWARE.equals("qcom")) {
            /**
             * Tensorflow model using NPU runtime can only be executed on 
             * Snapdragon. Cannot run this test on exynos.
             */
            return;
        }

        runSNAPTensorflow(APITestCommon.SNAPComputingUnit.NPU);
    }

    @Test
    public void testNNFWTFLite() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        File model = APITestCommon.getTFLiteAddModel();
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=nnfw model=" + model.getAbsolutePath() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{1,1,1,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    ByteBuffer buffer = data.getTensorData(0);
                    float expected = buffer.getFloat(0);

                    /* check received data */
                    if (expected != 3.5f) {
                        mInvalidState = true;
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            TensorsData input = info.allocate();

            ByteBuffer buffer = input.getTensorData(0);
            buffer.putFloat(0, 1.5f);

            input.setTensorData(0, buffer);

            pipe.inputData("srcx", input);

            /* sleep 1000 to invoke */
            Thread.sleep(1000);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }
}
