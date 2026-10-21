package com.thesis.braintumor.tflitebenchmark;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import android.content.res.AssetFileDescriptor;
import android.content.Context;
import android.util.Log;

public class TFLiteBenchmark {

    private Interpreter interpreter;
    private static final int IMG_SIZE = 640;
    private static final String TAG = "TFLiteBenchmark";

    public TFLiteBenchmark(Context context, String modelPath) throws IOException {
        MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(modelBuffer, options);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void runBenchmark(int numRuns, int warmupRuns) {
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * 3 * IMG_SIZE * IMG_SIZE * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        for (int i = 0; i < 3 * IMG_SIZE * IMG_SIZE; i++) {
            inputBuffer.putFloat((float) Math.random());
        }

        float[][][] output = new float[1][7][8400];

        for (int i = 0; i < warmupRuns; i++) {
            inputBuffer.rewind();
            interpreter.run(inputBuffer, output);
        }

        long[] timings = new long[numRuns];
        for (int i = 0; i < numRuns; i++) {
            inputBuffer.rewind();
            long start = System.nanoTime();
            interpreter.run(inputBuffer, output);
            long end = System.nanoTime();
            timings[i] = (end - start) / 1_000_000;
            Log.i(TAG, "Run " + i + ": " + timings[i] + " ms");
        }

        double mean = 0;
        for (long t : timings) mean += t;
        mean /= numRuns;

        double variance = 0;
        for (long t : timings) variance += Math.pow(t - mean, 2);
        double std = Math.sqrt(variance / numRuns);

        java.util.Arrays.sort(timings);
        long p95 = timings[(int) (numRuns * 0.95)];
        long min = timings[0];
        long max = timings[numRuns - 1];


        Log.i(TAG, String.format(
                "Latence -- mean: %.2f ms | std: %.2f ms | min: %d ms | max: %d ms | p95: %d ms | FPS: %.2f",
                mean, std, min, max, p95, 1000.0 / mean
        ));
    }

    public void close() {
        if (interpreter != null) interpreter.close();
    }
}
