package com.thesis.braintumor.tflitebenchmark;

import android.app.Activity;
import android.app.ActivityManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.os.SystemClock;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TensorFlowLite;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.ToLongFunction;

public class MainActivity extends Activity {

    // ================================================================
    // CONFIGURATION DU BENCHMARK
    // ================================================================

    private static final String[] MODEL_NAMES = {
            "models/adopted_tflite.tflite",
            "models/yolov12n_tflite.tflite"
    };

    private static final String IMAGE_FOLDER =
            "benchmark_images";

    // Même protocole que la mesure CPU sur Kaggle
    private static final int NUM_THREADS = 4;

    // Exécutions non comptabilisées (chaîne complète, 10 premières images)
    private static final int WARMUP_RUNS = 10;

    // Nombre d'images mesurées par exécution
    private static final int MAX_MEASURED_IMAGES = 100;

    // Ordre alterné A, B, A, B, A, B
    private static final int ROUNDS_PER_MODEL = 3;

    // Pause entre deux exécutions, pour limiter l'échauffement
    private static final long PAUSE_BETWEEN_RUNS_MS = 60_000L;

    // Mesure mémoire, toujours hors des sections chronométrées (voir runOneModel)
    private static final long MEMORY_SETTLE_MS = 500L;

    // Nouveaux fichiers : les résultats de latence précédents ne sont pas écrasés
    private static final String RESULT_JSON = "r1_8_phone_latency_memory.json";
    private static final String RESULT_SUMMARY = "r1_8_phone_summary_memory.txt";

    // Post-traitement identique à Ultralytics predict (conf, iou, max_det, max_nms)
    private static final float CONF_THRESHOLD = 0.25f;
    private static final float IOU_THRESHOLD = 0.7f;
    private static final int MAX_DETECTIONS = 300;
    private static final int MAX_NMS_CANDIDATES = 30000;

    private TextView outputText;

    // ================================================================
    // ON CREATE
    // ================================================================

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        outputText = new TextView(this);

        outputText.setTextSize(14f);

        outputText.setPadding(
                30,
                30,
                30,
                30
        );

        outputText.setText(
                "Initialisation du benchmark..."
        );

        ScrollView scrollView =
                new ScrollView(this);

        scrollView.addView(outputText);

        setContentView(scrollView);

        // Écran allumé pendant toute la mesure
        getWindow().addFlags(
                android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        );

        // Le benchmark ne doit pas tourner sur le thread UI.
        new Thread(() -> {

            try {

                String result =
                        runBenchmark();

                runOnUiThread(() ->
                        outputText.setText(result)
                );

            } catch (Exception e) {

                e.printStackTrace();

                String error =
                        "ERREUR\n\n"
                                + getStackTrace(e);

                runOnUiThread(() ->
                        outputText.setText(error)
                );
            }

        }).start();
    }

    // ================================================================
    // BENCHMARK PRINCIPAL
    // Les deux modèles, en alternance, chaîne complète chronométrée :
    // décodage image + prétraitement + inférence + post-traitement (NMS)
    // ================================================================

    private String runBenchmark()
            throws Exception {

        List<String> imageNames = listBenchmarkImages();

        // Bibliothèque native TFLite chargée avant la première mesure mémoire :
        // toutes les exécutions partent du même état.
        String tfliteRuntime = TensorFlowLite.runtimeVersion();
        long totalRamMb = totalRamMb();

        File resultDir = getExternalFilesDir(null);
        if (resultDir == null) {
            resultDir = getFilesDir();
        }

        List<RunResult> runs = new ArrayList<>();
        int totalRuns = ROUNDS_PER_MODEL * MODEL_NAMES.length;

        for (int round = 1; round <= ROUNDS_PER_MODEL; round++) {
            for (String modelName : MODEL_NAMES) {
                if (!runs.isEmpty()) {
                    status(String.format(Locale.US,
                            "Pause de %d s avant l'exécution %d/%d...",
                            PAUSE_BETWEEN_RUNS_MS / 1000, runs.size() + 1, totalRuns));
                    Thread.sleep(PAUSE_BETWEEN_RUNS_MS);
                }
                status(String.format(Locale.US,
                        "Exécution %d/%d : %s (tour %d) en cours...",
                        runs.size() + 1, totalRuns, modelName, round));
                runs.add(runOneModel(modelName, round, imageNames));
            }
        }

        // ============================================================
        // RESULTATS DETAILLES (JSON)
        // ============================================================

        JSONObject json = new JSONObject();
        json.put("device", getDeviceInfo());
        json.put("threads", NUM_THREADS);
        json.put("warmup_runs", WARMUP_RUNS);
        json.put("rounds_per_model", ROUNDS_PER_MODEL);
        json.put("pause_between_runs_ms", PAUSE_BETWEEN_RUNS_MS);
        json.put("conf_threshold", (double) CONF_THRESHOLD);
        json.put("iou_threshold", (double) IOU_THRESHOLD);
        json.put("max_detections", MAX_DETECTIONS);
        json.put("tflite_runtime", tfliteRuntime);
        json.put("device_total_ram_mb", totalRamMb);
        json.put("memory_protocol", "Outside the timed sections. Baseline before model loading (after GC); "
                + "snapshots after interpreter creation, after warm-up (after GC) and at the end; RSS after each "
                + "measured image; VmHWM reset at run start when allowed. Values in KiB.");
        JSONArray runsJson = new JSONArray();
        for (RunResult run : runs) {
            runsJson.put(run.toJson());
        }
        json.put("runs", runsJson);

        File jsonFile = new File(resultDir, RESULT_JSON);
        BufferedWriter jsonWriter = new BufferedWriter(new FileWriter(jsonFile));
        jsonWriter.write(json.toString(1));
        jsonWriter.close();

        // ============================================================
        // RESUME PAR MODELE (exécutions regroupées)
        // ============================================================

        StringBuilder summary = new StringBuilder();
        summary.append("\n========================================\n")
                .append("ANDROID TFLITE LATENCY AND MEMORY BENCHMARK\n")
                .append("========================================\n\n")
                .append("DEVICE\n------\n").append(getDeviceInfo())
                .append("RAM total    : ").append(totalRamMb).append(" MB\n")
                .append("TFLite runtime : ").append(tfliteRuntime).append("\n\n")
                .append("CONFIGURATION\n-------------\n")
                .append("CPU threads      : ").append(NUM_THREADS).append("\n")
                .append("Warm-up runs     : ").append(WARMUP_RUNS).append("\n")
                .append("Images per run   : ").append(imageNames.size()).append("\n")
                .append("Runs per model   : ").append(ROUNDS_PER_MODEL)
                .append(" (alternating order, ").append(PAUSE_BETWEEN_RUNS_MS / 1000).append(" s pauses)\n")
                .append("Post-processing  : conf > ").append(format(CONF_THRESHOLD))
                .append(", class-wise NMS IoU ").append(format(IOU_THRESHOLD))
                .append(", max ").append(MAX_DETECTIONS).append("\n\n");

        for (String modelName : MODEL_NAMES) {
            List<Double> decode = new ArrayList<>();
            List<Double> preprocessing = new ArrayList<>();
            List<Double> inference = new ArrayList<>();
            List<Double> postprocessing = new ArrayList<>();
            List<Double> total = new ArrayList<>();
            List<String> roundMeans = new ArrayList<>();
            double detections = 0.0;
            int[] inputShape = null;
            List<MemoryRecord> memories = new ArrayList<>();
            for (RunResult run : runs) {
                if (!run.modelName.equals(modelName)) {
                    continue;
                }
                inputShape = run.inputShape;
                memories.add(run.memory);
                double roundSum = 0.0;
                for (ResultRow row : run.rows) {
                    decode.add(row.decodeMs);
                    preprocessing.add(row.preprocessMs);
                    inference.add(row.inferenceMs);
                    postprocessing.add(row.postprocessMs);
                    total.add(row.totalMs);
                    detections += row.detections;
                    roundSum += row.totalMs;
                }
                roundMeans.add(format(roundSum / run.rows.size()));
            }
            Stats totalStats = calculateStats(total);
            Stats inferenceStats = calculateStats(inference);
            summary.append("MODEL ").append(modelName).append("\n")
                    .append("Input            : ").append(Arrays.toString(inputShape)).append("\n")
                    .append("Measurements     : ").append(total.size()).append("\n")
                    .append("TOTAL (ms)       : mean ").append(format(totalStats.mean))
                    .append(" | SD ").append(format(totalStats.sd))
                    .append(" | median ").append(format(totalStats.median))
                    .append(" | min ").append(format(totalStats.min))
                    .append(" | P95 ").append(format(totalStats.p95))
                    .append(" | max ").append(format(totalStats.max)).append("\n")
                    .append("Images/s         : ").append(format(1000.0 / totalStats.mean)).append("\n")
                    .append("Round means (ms) : ").append(roundMeans).append("\n")
                    .append("  decode         : ").append(format(calculateStats(decode).mean)).append(" ms\n")
                    .append("  preprocessing  : ").append(format(calculateStats(preprocessing).mean)).append(" ms\n")
                    .append("  inference      : ").append(format(inferenceStats.mean))
                    .append(" ms (SD ").append(format(inferenceStats.sd))
                    .append(", P95 ").append(format(inferenceStats.p95)).append(")\n")
                    .append("  postprocessing : ").append(format(calculateStats(postprocessing).mean)).append(" ms\n")
                    .append("Detections/image : ").append(format(detections / total.size())).append("\n");
            summary.append(memorySummary(memories)).append("\n");
        }

        summary.append("RESULT FILES\n------------\n")
                .append(jsonFile.getAbsolutePath()).append("\n");
        File summaryFile = new File(resultDir, RESULT_SUMMARY);
        summary.append(summaryFile.getAbsolutePath()).append("\n")
                .append("========================================");

        BufferedWriter summaryWriter = new BufferedWriter(new FileWriter(summaryFile));
        summaryWriter.write(summary.toString());
        summaryWriter.close();

        return summary.toString();
    }

    // ================================================================
    // UNE EXECUTION : un modèle, 10 warm-up, 100 images mesurées
    // ================================================================

    private RunResult runOneModel(
            String modelName,
            int round,
            List<String> imageNames
    ) throws Exception {

        // Mémoire : état de référence de l'exécution, avant tout chargement (hors chronométrage)
        MemoryRecord memory = new MemoryRecord();
        memory.base = memorySnapshot(true);
        memory.peakResetOk = resetPeakRss();

        byte[] modelBytes = readAssetBytes(modelName);
        ByteBuffer modelBuffer = ByteBuffer
                .allocateDirect(modelBytes.length)
                .order(ByteOrder.nativeOrder());
        modelBuffer.put(modelBytes);
        modelBuffer.rewind();
        modelBytes = null;  // copié dans modelBuffer ; libéré avant les mesures mémoire

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(NUM_THREADS);
        Interpreter interpreter = new Interpreter(modelBuffer, options);

        try {
            memory.loaded = memorySnapshot(false);  // modèle chargé, tenseurs alloués
            int[] inputShape = interpreter.getInputTensor(0).shape();
            if (interpreter.getInputTensor(0).dataType() != DataType.FLOAT32
                    || inputShape.length != 4) {
                throw new IllegalStateException(
                        "Entrée inattendue : " + Arrays.toString(inputShape));
            }
            boolean isNCHW = inputShape[1] == 3;
            int inputHeight = isNCHW ? inputShape[2] : inputShape[1];
            int inputWidth = isNCHW ? inputShape[3] : inputShape[2];

            // Sortie YOLO : [1, 4 + nombre de classes, ancres]
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            if (interpreter.getOutputTensorCount() != 1 || outputShape.length != 3) {
                throw new IllegalStateException(
                        "Sortie inattendue : " + Arrays.toString(outputShape));
            }

            Map<Integer, Object> outputs = createOutputBuffers(interpreter);

            // Warm-up : chaîne complète sur les premières images, non comptabilisée
            for (int i = 0; i < WARMUP_RUNS; i++) {
                String name = imageNames.get(i % imageNames.size());
                Bitmap bitmap = loadBitmap(IMAGE_FOLDER + "/" + name);
                ByteBuffer input = preprocess(bitmap, inputWidth, inputHeight, isNCHW);
                bitmap.recycle();
                resetOutputBuffers(outputs);
                interpreter.runForMultipleInputsOutputs(new Object[]{input}, outputs);
                postprocess((ByteBuffer) outputs.get(0), outputShape, inputWidth, inputHeight);
            }

            memory.warm = memorySnapshot(true);  // régime établi après warm-up (hors chronométrage)

            // Mesures
            List<ResultRow> rows = new ArrayList<>();
            for (String name : imageNames) {
                long t0 = SystemClock.elapsedRealtimeNanos();
                Bitmap bitmap = loadBitmap(IMAGE_FOLDER + "/" + name);
                long t1 = SystemClock.elapsedRealtimeNanos();
                ByteBuffer input = preprocess(bitmap, inputWidth, inputHeight, isNCHW);
                long t2 = SystemClock.elapsedRealtimeNanos();
                resetOutputBuffers(outputs);
                interpreter.runForMultipleInputsOutputs(new Object[]{input}, outputs);
                long t3 = SystemClock.elapsedRealtimeNanos();
                List<Detection> detections = postprocess(
                        (ByteBuffer) outputs.get(0), outputShape, inputWidth, inputHeight);
                long t4 = SystemClock.elapsedRealtimeNanos();
                bitmap.recycle();
                rows.add(new ResultRow(
                        name,
                        nanosToMs(t1 - t0),
                        nanosToMs(t2 - t1),
                        nanosToMs(t3 - t2),
                        nanosToMs(t4 - t3),
                        nanosToMs(t4 - t0),
                        detections.size()));
                // RSS relevé après la mesure de l'image, hors chronométrage
                memory.maxSampledRssKb = Math.max(memory.maxSampledRssKb, readProcStatusKb("VmRSS"));
            }
            memory.end = memorySnapshot(false);
            RunResult result = new RunResult(modelName, round, inputShape, outputShape, rows);
            result.memory = memory;
            return result;
        } finally {
            interpreter.close();
        }
    }

    // ================================================================
    // POST-TRAITEMENT YOLO (identique à Ultralytics non_max_suppression)
    // Sortie [1, 4 + nc, ancres] : cx, cy, w, h normalisés à [0, 1],
    // puis scores de classe (sigmoïde déjà appliquée dans le modèle).
    // ================================================================

    private List<Detection> postprocess(
            ByteBuffer output,
            int[] shape,
            int inputWidth,
            int inputHeight
    ) {
        int channels = shape[1];
        int anchors = shape[2];
        int numClasses = channels - 4;

        output.rewind();
        FloatBuffer values = output.asFloatBuffer();

        List<Detection> candidates = new ArrayList<>();
        for (int a = 0; a < anchors; a++) {
            int bestClass = -1;
            float bestScore = 0f;
            for (int c = 0; c < numClasses; c++) {
                float score = values.get((4 + c) * anchors + a);
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }
            if (bestScore > CONF_THRESHOLD) {
                float cx = values.get(a) * inputWidth;
                float cy = values.get(anchors + a) * inputHeight;
                float w = values.get(2 * anchors + a) * inputWidth;
                float h = values.get(3 * anchors + a) * inputHeight;
                candidates.add(new Detection(
                        cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f,
                        bestScore, bestClass));
            }
        }

        // Tri par confiance décroissante, au plus MAX_NMS_CANDIDATES
        Collections.sort(candidates, (p, q) -> Float.compare(q.score, p.score));
        if (candidates.size() > MAX_NMS_CANDIDATES) {
            candidates = new ArrayList<>(candidates.subList(0, MAX_NMS_CANDIDATES));
        }

        // NMS par classe : suppression si IoU > seuil avec une boîte retenue de même classe
        List<Detection> kept = new ArrayList<>();
        boolean[] removed = new boolean[candidates.size()];
        for (int i = 0; i < candidates.size() && kept.size() < MAX_DETECTIONS; i++) {
            if (removed[i]) {
                continue;
            }
            Detection keptBox = candidates.get(i);
            kept.add(keptBox);
            for (int j = i + 1; j < candidates.size(); j++) {
                if (!removed[j]
                        && candidates.get(j).cls == keptBox.cls
                        && iou(keptBox, candidates.get(j)) > IOU_THRESHOLD) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    private static float iou(Detection a, Detection b) {
        float w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
        float h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
        if (w <= 0f || h <= 0f) {
            return 0f;
        }
        float intersection = w * h;
        return intersection / (a.area() + b.area() - intersection);
    }

    // ================================================================
    // MESURE MEMOIRE (toujours hors chronométrage)
    // PSS, tas natif et tas Java : android.os.Debug / Runtime ;
    // RSS et pic de RSS (VmHWM) : /proc/self/status
    // ================================================================

    private MemorySnapshot memorySnapshot(boolean settle) {
        if (settle) {
            System.gc();
            SystemClock.sleep(50L);
            System.gc();
            SystemClock.sleep(MEMORY_SETTLE_MS);
        }
        Debug.MemoryInfo info = new Debug.MemoryInfo();
        Debug.getMemoryInfo(info);
        Runtime runtime = Runtime.getRuntime();
        return new MemorySnapshot(
                info.getTotalPss(),
                readProcStatusKb("VmRSS"),
                readProcStatusKb("VmHWM"),
                Debug.getNativeHeapAllocatedSize() / 1024L,
                (runtime.totalMemory() - runtime.freeMemory()) / 1024L);
    }

    private static long readProcStatusKb(String key) {
        try (BufferedReader reader = new BufferedReader(new FileReader("/proc/self/status"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith(key + ":")) {
                    return Long.parseLong(line.substring(key.length() + 1).trim().split("\\s+")[0]);
                }
            }
        } catch (Exception ignored) {
        }
        return -1L;
    }

    // Remet à zéro le pic de RSS du processus (VmHWM) ; false si le système le refuse
    private static boolean resetPeakRss() {
        try (FileWriter writer = new FileWriter("/proc/self/clear_refs")) {
            writer.write("5");
        } catch (Exception e) {
            return false;
        }
        long hwm = readProcStatusKb("VmHWM");
        long rss = readProcStatusKb("VmRSS");
        return hwm > 0 && rss > 0 && hwm <= rss + 1024L;
    }

    private long totalRamMb() {
        ActivityManager manager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        manager.getMemoryInfo(info);
        return info.totalMem / (1024L * 1024L);
    }

    private String memorySummary(List<MemoryRecord> records) {
        boolean resetOk = true;
        for (MemoryRecord r : records) {
            resetOk &= r.peakResetOk;
        }
        return "Memory (MiB, increase over the pre-loading baseline of each run;\n"
                + "        median of the runs, run values in brackets)\n"
                + memoryLine("  PSS after model loading    : ", records, r -> r.loaded.pssKb - r.base.pssKb)
                + memoryLine("  PSS after warm-up (GC)     : ", records, r -> r.warm.pssKb - r.base.pssKb)
                + memoryLine("  RSS after warm-up (GC)     : ", records, r -> r.warm.rssKb - r.base.rssKb)
                + memoryLine("  native heap after warm-up  : ", records, r -> r.warm.nativeAllocKb - r.base.nativeAllocKb)
                + memoryLine("  Java heap after warm-up    : ", records, r -> r.warm.javaUsedKb - r.base.javaUsedKb)
                + memoryLine("  peak RSS during the run    : ", records, r -> r.peakRssKb() < 0 ? Long.MIN_VALUE : r.peakRssKb() - r.base.rssKb)
                + memoryLine("  baseline PSS (absolute)    : ", records, r -> r.base.pssKb)
                + "  peak RSS source            : " + (resetOk ? "VmHWM reset per run + RSS after each image" : "RSS after each image (VmHWM reset not allowed)") + "\n";
    }

    private String memoryLine(String label, List<MemoryRecord> records, ToLongFunction<MemoryRecord> valueKb) {
        List<Double> values = new ArrayList<>();
        List<String> shown = new ArrayList<>();
        for (MemoryRecord r : records) {
            long kb = valueKb.applyAsLong(r);
            if (kb == Long.MIN_VALUE) {
                shown.add("n/a");
                continue;
            }
            values.add(kb / 1024.0);
            shown.add(format(kb / 1024.0));
        }
        String median = values.isEmpty() ? "n/a" : format(calculateStats(values).median);
        return label + median + "  " + shown + "\n";
    }

    // ================================================================
    // LISTE DES IMAGES DE BENCHMARK (ordre alphabétique)
    // ================================================================

    private List<String> listBenchmarkImages()
            throws Exception {

        String[] files = getAssets().list(IMAGE_FOLDER);
        if (files == null || files.length == 0) {
            throw new IllegalStateException(
                    "Aucune image trouvée dans : assets/" + IMAGE_FOLDER + "/");
        }
        Arrays.sort(files);
        List<String> imageNames = new ArrayList<>();
        for (String file : files) {
            String lower = file.toLowerCase(Locale.US);
            if (lower.endsWith(".png") || lower.endsWith(".jpg")
                    || lower.endsWith(".jpeg") || lower.endsWith(".bmp")) {
                imageNames.add(file);
                if (imageNames.size() >= MAX_MEASURED_IMAGES) {
                    break;
                }
            }
        }
        if (imageNames.size() < WARMUP_RUNS) {
            throw new IllegalStateException(
                    "Trop peu d'images : " + imageNames.size());
        }
        return imageNames;
    }

    // ================================================================
    // AFFICHAGE DE L'AVANCEMENT
    // ================================================================

    private void status(String message) {
        System.out.println(message);
        runOnUiThread(() -> outputText.setText(message));
    }

    // ================================================================
    // PREPROCESSING YOLO
    // ================================================================

    private ByteBuffer preprocess(
            Bitmap source,
            int targetWidth,
            int targetHeight,
            boolean nchw
    ) {

        // ============================================================
        // LETTERBOX
        // ============================================================

        float scale =
                Math.min(
                        (float) targetWidth
                                / source.getWidth(),

                        (float) targetHeight
                                / source.getHeight()
                );

        int resizedWidth =
                Math.round(
                        source.getWidth()
                                * scale
                );

        int resizedHeight =
                Math.round(
                        source.getHeight()
                                * scale
                );

        float offsetX =
                (
                        targetWidth
                                - resizedWidth
                ) / 2.0f;

        float offsetY =
                (
                        targetHeight
                                - resizedHeight
                ) / 2.0f;

        Bitmap letterboxed =
                Bitmap.createBitmap(
                        targetWidth,
                        targetHeight,
                        Bitmap.Config.ARGB_8888
                );

        Canvas canvas =
                new Canvas(
                        letterboxed
                );

        // Couleur standard utilisée par Ultralytics
        // pour le letterbox.
        canvas.drawColor(
                Color.rgb(
                        114,
                        114,
                        114
                )
        );

        Paint paint =
                new Paint(
                        Paint.ANTI_ALIAS_FLAG
                                | Paint.FILTER_BITMAP_FLAG
                );

        RectF destination =
                new RectF(
                        offsetX,
                        offsetY,
                        offsetX + resizedWidth,
                        offsetY + resizedHeight
                );

        canvas.drawBitmap(
                source,
                null,
                destination,
                paint
        );

        // ============================================================
        // EXTRACTION PIXELS
        // ============================================================

        int[] pixels =
                new int[
                        targetWidth
                                * targetHeight
                        ];

        letterboxed.getPixels(
                pixels,
                0,
                targetWidth,
                0,
                0,
                targetWidth,
                targetHeight
        );

        // Float32 :
        //
        // width * height * 3 canaux * 4 octets
        //
        ByteBuffer buffer =
                ByteBuffer
                        .allocateDirect(
                                targetWidth
                                        * targetHeight
                                        * 3
                                        * 4
                        )
                        .order(
                                ByteOrder.nativeOrder()
                        );

        // ============================================================
        // NCHW
        // ============================================================

        if (nchw) {

            // Canal R
            for (int pixel : pixels) {

                float r =
                        Color.red(pixel)
                                / 255.0f;

                buffer.putFloat(r);
            }

            // Canal G
            for (int pixel : pixels) {

                float g =
                        Color.green(pixel)
                                / 255.0f;

                buffer.putFloat(g);
            }

            // Canal B
            for (int pixel : pixels) {

                float b =
                        Color.blue(pixel)
                                / 255.0f;

                buffer.putFloat(b);
            }

        }

        // ============================================================
        // NHWC
        // ============================================================

        else {

            for (int pixel : pixels) {

                float r =
                        Color.red(pixel)
                                / 255.0f;

                float g =
                        Color.green(pixel)
                                / 255.0f;

                float b =
                        Color.blue(pixel)
                                / 255.0f;

                buffer.putFloat(r);

                buffer.putFloat(g);

                buffer.putFloat(b);
            }
        }

        buffer.rewind();

        letterboxed.recycle();

        return buffer;
    }

    // ================================================================
    // CREATION DES OUTPUT BUFFERS
    // ================================================================

    private Map<Integer, Object>
    createOutputBuffers(
            Interpreter interpreter
    ) {

        Map<Integer, Object> outputs =
                new HashMap<>();

        for (
                int i = 0;
                i < interpreter.getOutputTensorCount();
                i++
        ) {

            int numBytes =
                    interpreter
                            .getOutputTensor(i)
                            .numBytes();

            ByteBuffer buffer =
                    ByteBuffer
                            .allocateDirect(
                                    numBytes
                            )
                            .order(
                                    ByteOrder.nativeOrder()
                            );

            outputs.put(
                    i,
                    buffer
            );
        }

        return outputs;
    }

    // ================================================================
    // RESET OUTPUT BUFFERS
    // ================================================================

    private void resetOutputBuffers(
            Map<Integer, Object> outputs
    ) {

        for (
                Object output :
                outputs.values()
        ) {

            if (
                    output
                            instanceof ByteBuffer
            ) {

                ByteBuffer buffer =
                        (ByteBuffer) output;

                buffer.rewind();
            }
        }
    }

    // ================================================================
    // CHARGEMENT D'UNE IMAGE DEPUIS ASSETS
    // ================================================================

    private Bitmap loadBitmap(
            String assetPath
    ) throws Exception {

        InputStream input =
                null;

        try {

            input =
                    getAssets()
                            .open(
                                    assetPath
                            );

            Bitmap bitmap =
                    BitmapFactory
                            .decodeStream(
                                    input
                            );

            if (bitmap == null) {

                throw new IllegalStateException(
                        "Impossible de décoder l'image : "
                                + assetPath
                );
            }

            return bitmap;

        } finally {

            if (input != null) {

                try {

                    input.close();

                } catch (Exception ignored) {
                }
            }
        }
    }

    // ================================================================
    // CHARGEMENT DU MODELE DEPUIS ASSETS
    //
    // Version compatible Android sans readAllBytes()
    // ================================================================

    private byte[] readAssetBytes(
            String assetPath
    ) throws Exception {

        java.io.InputStream input =
                getAssets().open(
                        assetPath
                );

        java.io.ByteArrayOutputStream output =
                new java.io.ByteArrayOutputStream();

        byte[] buffer =
                new byte[8192];

        int length;

        while (
                (length = input.read(buffer)) != -1
        ) {

            output.write(
                    buffer,
                    0,
                    length
            );
        }

        input.close();

        return output.toByteArray();
    }

    // ================================================================
    // INFORMATIONS DE L'APPAREIL
    // ================================================================

    private String getDeviceInfo() {

        StringBuilder builder =
                new StringBuilder();

        builder.append(
                "Manufacturer : "
        );

        builder.append(
                Build.MANUFACTURER
        );

        builder.append("\n");

        builder.append(
                "Brand        : "
        );

        builder.append(
                Build.BRAND
        );

        builder.append("\n");

        builder.append(
                "Model        : "
        );

        builder.append(
                Build.MODEL
        );

        builder.append("\n");

        builder.append(
                "Device       : "
        );

        builder.append(
                Build.DEVICE
        );

        builder.append("\n");

        builder.append(
                "Hardware     : "
        );

        builder.append(
                Build.HARDWARE
        );

        builder.append("\n");

        // ============================================================
        // SOC
        // ============================================================

        if (
                Build.VERSION.SDK_INT
                        >= Build.VERSION_CODES.S
        ) {

            builder.append(
                    "SoC manufacturer : "
            );

            builder.append(
                    Build.SOC_MANUFACTURER
            );

            builder.append("\n");

            builder.append(
                    "SoC model        : "
            );

            builder.append(
                    Build.SOC_MODEL
            );

            builder.append("\n");
        }

        builder.append(
                "Android      : "
        );

        builder.append(
                Build.VERSION.RELEASE
        );

        builder.append("\n");

        builder.append(
                "SDK          : "
        );

        builder.append(
                Build.VERSION.SDK_INT
        );

        builder.append("\n");

        builder.append(
                "Security patch : "
        );

        builder.append(
                Build.VERSION.SECURITY_PATCH
        );

        builder.append("\n");

        builder.append(
                "CPU cores visible : "
        );

        builder.append(
                Runtime
                        .getRuntime()
                        .availableProcessors()
        );

        builder.append("\n");

        builder.append(
                "ABI : "
        );

        if (
                Build.SUPPORTED_ABIS != null
                        && Build.SUPPORTED_ABIS.length > 0
        ) {

            builder.append(
                    Arrays.toString(
                            Build.SUPPORTED_ABIS
                    )
            );

        } else {

            builder.append(
                    "Unknown"
            );
        }

        builder.append("\n");

        return builder.toString();
    }

    // ================================================================
    // STATISTIQUES
    // ================================================================

    private Stats calculateStats(
            List<Double> values
    ) {

        if (
                values == null
                        || values.isEmpty()
        ) {

            throw new IllegalArgumentException(
                    "Liste de valeurs vide."
            );
        }

        List<Double> sorted =
                new ArrayList<>(
                        values
                );

        Collections.sort(
                sorted
        );

        int n =
                sorted.size();

        // ============================================================
        // MOYENNE
        // ============================================================

        double sum =
                0.0;

        for (
                double value :
                sorted
        ) {

            sum += value;
        }

        double mean =
                sum / n;

        // ============================================================
        // ECART-TYPE ECHANTILLON
        // ============================================================

        double variance =
                0.0;

        if (n > 1) {

            for (
                    double value :
                    sorted
            ) {

                double difference =
                        value
                                - mean;

                variance +=
                        difference
                                * difference;
            }

            variance /=
                    (n - 1);
        }

        double sd =
                Math.sqrt(
                        variance
                );

        // ============================================================
        // MEDIANE
        // ============================================================

        double median;

        if (n % 2 == 0) {

            median =
                    (
                            sorted.get(
                                    n / 2 - 1
                            )
                                    +
                                    sorted.get(
                                            n / 2
                                    )
                    ) / 2.0;

        } else {

            median =
                    sorted.get(
                            n / 2
                    );
        }

        // ============================================================
        // P95
        // ============================================================

        int p95Index =
                (int) Math.ceil(
                        0.95
                                * n
                ) - 1;

        if (p95Index < 0) {

            p95Index =
                    0;
        }

        if (p95Index >= n) {

            p95Index =
                    n - 1;
        }

        // ============================================================
        // RESULTAT
        // ============================================================

        return new Stats(
                mean,
                sd,
                sorted.get(0),
                median,
                sorted.get(p95Index),
                sorted.get(n - 1)
        );
    }

    // ================================================================
    // NANOSECONDES -> MILLISECONDES
    // ================================================================

    private double nanosToMs(
            long nanos
    ) {

        return nanos
                / 1_000_000.0;
    }

    // ================================================================
    // FORMAT
    // ================================================================

    private String format(
            double value
    ) {

        return String.format(
                Locale.US,
                "%.2f",
                value
        );
    }

    // ================================================================
    // STACK TRACE
    // ================================================================

    private String getStackTrace(
            Exception exception
    ) {

        StringBuilder builder =
                new StringBuilder();

        builder
                .append(
                        exception.toString()
                )
                .append(
                        "\n\n"
                );

        for (
                StackTraceElement element :
                exception.getStackTrace()
        ) {

            builder
                    .append(
                            element.toString()
                    )
                    .append(
                            "\n"
                    );
        }

        return builder.toString();
    }

    // ================================================================
    // RESULT ROW
    // ================================================================

    private static class ResultRow {

        final String image;
        final double decodeMs;
        final double preprocessMs;
        final double inferenceMs;
        final double postprocessMs;
        final double totalMs;
        final int detections;

        ResultRow(
                String image,
                double decodeMs,
                double preprocessMs,
                double inferenceMs,
                double postprocessMs,
                double totalMs,
                int detections
        ) {
            this.image = image;
            this.decodeMs = decodeMs;
            this.preprocessMs = preprocessMs;
            this.inferenceMs = inferenceMs;
            this.postprocessMs = postprocessMs;
            this.totalMs = totalMs;
            this.detections = detections;
        }
    }

    private static class Detection {

        final float x1;
        final float y1;
        final float x2;
        final float y2;
        final float score;
        final int cls;

        Detection(float x1, float y1, float x2, float y2, float score, int cls) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.cls = cls;
        }

        float area() {
            return Math.max(0f, x2 - x1) * Math.max(0f, y2 - y1);
        }
    }

    private static class RunResult {

        final String modelName;
        final int round;
        final int[] inputShape;
        final int[] outputShape;
        final List<ResultRow> rows;
        MemoryRecord memory;

        RunResult(String modelName, int round, int[] inputShape, int[] outputShape, List<ResultRow> rows) {
            this.modelName = modelName;
            this.round = round;
            this.inputShape = inputShape;
            this.outputShape = outputShape;
            this.rows = rows;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("model", modelName);
            json.put("round", round);
            json.put("input_shape", Arrays.toString(inputShape));
            json.put("output_shape", Arrays.toString(outputShape));
            JSONArray images = new JSONArray();
            for (ResultRow row : rows) {
                JSONObject item = new JSONObject();
                item.put("image", row.image);
                item.put("decode_ms", row.decodeMs);
                item.put("preprocess_ms", row.preprocessMs);
                item.put("inference_ms", row.inferenceMs);
                item.put("postprocess_ms", row.postprocessMs);
                item.put("total_ms", row.totalMs);
                item.put("detections", row.detections);
                images.put(item);
            }
            json.put("images", images);
            if (memory != null) {
                json.put("memory", memory.toJson());
            }
            return json;
        }
    }

    private static class MemorySnapshot {

        final long pssKb;
        final long rssKb;
        final long hwmKb;
        final long nativeAllocKb;
        final long javaUsedKb;

        MemorySnapshot(long pssKb, long rssKb, long hwmKb, long nativeAllocKb, long javaUsedKb) {
            this.pssKb = pssKb;
            this.rssKb = rssKb;
            this.hwmKb = hwmKb;
            this.nativeAllocKb = nativeAllocKb;
            this.javaUsedKb = javaUsedKb;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("pss_kb", pssKb);
            json.put("rss_kb", rssKb);
            json.put("vmhwm_kb", hwmKb);
            json.put("native_heap_allocated_kb", nativeAllocKb);
            json.put("java_heap_used_kb", javaUsedKb);
            return json;
        }
    }

    private static class MemoryRecord {

        MemorySnapshot base;
        MemorySnapshot loaded;
        MemorySnapshot warm;
        MemorySnapshot end;
        long maxSampledRssKb = -1L;
        boolean peakResetOk;

        // Pic de RSS de l'exécution : VmHWM (si remis à zéro au début) et RSS relevé après chaque image
        long peakRssKb() {
            long peak = Math.max(maxSampledRssKb, peakResetOk ? end.hwmKb : -1L);
            return peak > 0 ? peak : -1L;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("before_loading", base.toJson());
            json.put("after_loading", loaded.toJson());
            json.put("after_warmup", warm.toJson());
            json.put("end_of_run", end.toJson());
            json.put("max_rss_after_each_image_kb", maxSampledRssKb);
            json.put("vmhwm_reset_ok", peakResetOk);
            json.put("peak_rss_kb", peakRssKb());
            return json;
        }
    }

    // ================================================================
    // STATS
    // ================================================================

    private static class Stats {

        double mean;

        double sd;

        double min;

        double median;

        double p95;

        double max;

        Stats(
                double mean,
                double sd,
                double min,
                double median,
                double p95,
                double max
        ) {

            this.mean =
                    mean;

            this.sd =
                    sd;

            this.min =
                    min;

            this.median =
                    median;

            this.p95 =
                    p95;

            this.max =
                    max;
        }
    }
}