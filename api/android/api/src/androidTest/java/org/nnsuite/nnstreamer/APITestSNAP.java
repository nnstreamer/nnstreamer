package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Testcases for the filter sub-plugin SNAP.
 * Note that, it takes a time with GPU option in the first run.
 */
@RunWith(AndroidJUnit4.class)
public class APITestSNAP {
    private int mReceived = 0;
    private boolean mInvalidState = false;
    private boolean mIsAvailable = false;

    /**
     * Gets the File objects of SNAP Caffe model.
     */
    private File[] getCaffeModel() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();

        File model = new File(root + "/nnstreamer/snap_data/prototxt/squeezenet.prototxt");
        File weight = new File(root + "/nnstreamer/snap_data/model/squeezenet.caffemodel");

        if (!model.exists() || !weight.exists()) {
            fail();
        }

        return new File[]{model, weight};
    }

    /**
     * Gets the option string to run Caffe model.
     *
     * CPU: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:CPU"
     * GPU: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:GPU,GpuCacheSource:/sdcard/nnstreamer/"
     */
    private String getCaffeOption(boolean useGPU) {
        String option = "ModelFWType:CAFFE,ExecutionDataType:FLOAT32,";

        if (useGPU) {
            String root = Environment.getExternalStorageDirectory().getAbsolutePath();
            option = option + "ComputingUnit:GPU,GpuCacheSource:" + root + "/nnstreamer/";
        } else {
            option = option + "ComputingUnit:CPU";
        }

        return option;
    }

    /**
     * SingleShot with option string.
     */
    private void runSingleShotCaffe(boolean useGPU) {
        File[] models = getCaffeModel();
        String option = getCaffeOption(useGPU);

        try {
            TensorsInfo in = new TensorsInfo();
            in.addTensorInfo("data", NNStreamer.TensorType.FLOAT32, new int[]{3,224,224,1});

            TensorsInfo out = new TensorsInfo();
            out.addTensorInfo("prob", NNStreamer.TensorType.FLOAT32, new int[]{1,1,1000,1});

            SingleShot single = new SingleShot(models, in, out, NNStreamer.NNFWType.SNAP, option);

            /* let's ignore timeout (set 60 sec) */
            single.setTimeout(60000);

            /* single-shot invoke */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                TensorsData output = single.invoke(in.allocate());

                /* output: float32 1:1:1000:1 (NCHW format) */
                assertEquals(1, output.getTensorsCount());
                assertEquals(4000, output.getTensorData(0).capacity());

                Thread.sleep(30);
            }
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Pipeline with option string.
     */
    private void runPipelineCaffe(boolean useGPU) {
        File[] models = getCaffeModel();
        String option = getCaffeOption(useGPU);

        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)float32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=snap " +
                    "model=" + models[0].getAbsolutePath() + "," + models[1].getAbsolutePath() + " " +
                    "input=3:224:224:1 inputtype=float32 inputname=data " +
                    "output=1:1:1000:1 outputtype=float32 outputname=prob " +
                    "custom=" + option + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
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

    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        mReceived = 0;
        mInvalidState = false;
        mIsAvailable = NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP);
    }

    @Test
    public void testRunSingleCaffeCPU() {
        if (!mIsAvailable) {
            /* cannot run the test */
            return;
        }

        runSingleShotCaffe(false);
    }

    @Test
    public void testRunSingleCaffeGPU() {
        if (!mIsAvailable) {
            /* cannot run the test */
            return;
        }

        runSingleShotCaffe(true);
    }

    @Test
    public void testRunPipelineCaffeCPU() {
        if (!mIsAvailable) {
            /* cannot run the test */
            return;
        }

        runPipelineCaffe(false);
    }

    @Test
    public void testRunPipelineCaffeGPU() {
        if (!mIsAvailable) {
            /* cannot run the test */
            return;
        }

        runPipelineCaffe(true);
    }
}
