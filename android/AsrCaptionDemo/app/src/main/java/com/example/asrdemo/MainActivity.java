package com.example.asrdemo;

import android.Manifest;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.app.ActivityManager;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Process;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.TextUtils;
import android.text.style.ForegroundColorSpan;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.k2fsa.sherpa.onnx.EndpointConfig;
import com.k2fsa.sherpa.onnx.EndpointRule;
import com.k2fsa.sherpa.onnx.FeatureConfig;
import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MainActivity extends Activity {
    private static final int SAMPLE_RATE = 16000;
    private static final int REQ_PERM = 1;
    private static final int UI_CAPTION_EVERY = 2;
    private static final int UI_STATS_EVERY = 5;

    // 视图
    private TextView modelPill, statusText, timerText, captionBody, capTitle, liveBadge, statsText;
    private ImageView recordBtn;
    private View statusDot, ring;
    private WaveformView waveform;
    private ScrollView captionScroll;
    private TextView benchBtn;
    private LinearLayout stage;
    private ObjectAnimator dotAnim;
    private android.animation.Animator ringAnim;

    // 状态
    private volatile boolean running = false;
    private volatile boolean benchRunning = false;
    private OnlineRecognizer recognizer;
    private String modelLabel = "加载中";
    private int numThreads = 4;
    private int chunkMs = 100;
    private String provider = "cpu";
    private final List<String> committed = new ArrayList<>();
    private float lastRms = 0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Window w = getWindow();
        w.setStatusBarColor(Ui.BG);
        w.setNavigationBarColor(Ui.BG);
        w.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        w.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        numThreads = getIntent().getIntExtra("threads", 4);
        chunkMs = getIntent().getIntExtra("chunkMs", 100);
        provider = getIntent().getStringExtra("provider");
        if (provider == null) provider = "cpu";

        setContentView(buildUi());
        updateIdleUi();

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_PERM);
        }
        new Thread(this::loadModel).start();
    }

    // ---------------- 界面搭建 ----------------

    private int d(float dp) { return Ui.dp(this, dp); }

    private TextView text(String s, float sp, int color, int style) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        if (style != 0) t.setTypeface(Typeface.create("sans-serif", style));
        return t;
    }

    private TextView mono(String s, float sp, int color) {
        TextView t = text(s, sp, color, 0);
        t.setTypeface(Typeface.MONOSPACE);
        t.setIncludeFontPadding(false);
        return t;
    }

    private View dashLine(int color, int heightDp) {
        View v = new View(this);
        v.setBackgroundColor(color);
        v.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, d(heightDp)));
        return v;
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(Ui.BG);
        scroll.setFillViewport(true);

        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(d(20), d(12), d(20), d(28));
        scroll.addView(col, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // ——— 页头：标题与模型标签同行，副标题独占一行 ———
        LinearLayout head = new LinearLayout(this);
        head.setOrientation(LinearLayout.VERTICAL);

        LinearLayout titleRow = new LinearLayout(this);
        titleRow.setOrientation(LinearLayout.HORIZONTAL);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView h1 = text("实时字幕", 24, Ui.FG, Typeface.BOLD);
        titleRow.addView(h1, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        modelPill = mono("加载中", 10.5f, Ui.ACCENT);
        modelPill.setSingleLine(true);
        modelPill.setEllipsize(TextUtils.TruncateAt.END);
        modelPill.setMaxWidth(d(186));
        modelPill.setBackground(Ui.round(this, Ui.ACCENT_SOFT, 999f));
        modelPill.setPadding(d(10), d(5), d(10), d(5));
        titleRow.addView(modelPill);
        head.addView(titleRow);

        TextView lead = text("边说边转写，实时上屏。", 13.5f, Ui.MUTED, 0);
        lead.setPadding(0, d(3), 0, 0);
        head.addView(lead);
        col.addView(head);

        // ——— 录音舞台 ———
        stage = new LinearLayout(this);
        stage.setOrientation(LinearLayout.VERTICAL);
        stage.setGravity(Gravity.CENTER_HORIZONTAL);
        stage.setPadding(0, d(18), 0, 0);

        FrameLayout ctr = new FrameLayout(this);
        LinearLayout.LayoutParams ctrLp = new LinearLayout.LayoutParams(d(132), d(132));
        ctrLp.topMargin = d(14);
        col.addView(stage);
        stage.addView(ctr, ctrLp);

        ring = new View(this);
        ring.setBackground(Ui.ovalRing(this, Ui.ACCENT_RING, 2f));
        ring.setAlpha(0f);
        ctr.addView(ring, new FrameLayout.LayoutParams(d(160), d(160), Gravity.CENTER));

        recordBtn = new ImageView(this);
        recordBtn.setImageResource(R.drawable.ic_mic);
        recordBtn.setScaleType(ImageView.ScaleType.CENTER);
        recordBtn.setClickable(true);
        recordBtn.setBackground(Ui.pressable(this, Ui.ACCENT, Ui.ACCENT_PRESSED, 999f));
        recordBtn.setElevation(d(6));
        recordBtn.setOnClickListener(v -> toggleMic());
        ctr.addView(recordBtn, new FrameLayout.LayoutParams(d(132), d(132), Gravity.CENTER));

        LinearLayout status = new LinearLayout(this);
        status.setOrientation(LinearLayout.HORIZONTAL);
        status.setGravity(Gravity.CENTER_VERTICAL);
        status.setPadding(0, d(18), 0, 0);
        statusDot = new View(this);
        statusDot.setBackground(Ui.oval(this, Ui.BORDER));
        status.addView(statusDot, new LinearLayout.LayoutParams(d(8), d(8)));
        statusText = text("点击麦克风开始识别", 14, Ui.MUTED, 0);
        LinearLayout.LayoutParams stLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        stLp.leftMargin = d(8);
        status.addView(statusText, stLp);
        stage.addView(status);

        timerText = mono("00:00", 44, Ui.FG);
        timerText.setLetterSpacing(-0.02f);
        LinearLayout.LayoutParams tLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        tLp.topMargin = d(8);
        stage.addView(timerText, tLp);

        waveform = new WaveformView(this, null);
        LinearLayout.LayoutParams wLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, d(56));
        wLp.topMargin = d(4);
        stage.addView(waveform, wLp);

        // ——— 转写卡 ———
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackground(Ui.roundStroke(this, Ui.SURFACE, Ui.R_LG, Ui.BORDER, 1f));
        card.setPadding(d(20), d(16), d(20), d(18));
        LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        cardLp.topMargin = d(18);
        col.addView(card, cardLp);

        LinearLayout capRow = new LinearLayout(this);
        capRow.setOrientation(LinearLayout.HORIZONTAL);
        capRow.setGravity(Gravity.CENTER_VERTICAL);
        capTitle = mono("实时转写", 11.5f, Ui.MUTED);
        capTitle.setLetterSpacing(0.06f);
        capRow.addView(capTitle, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        liveBadge = mono("待机", 11f, Ui.MUTED);
        liveBadge.setBackground(Ui.round(this, Ui.FG_SOFT, 999f));
        liveBadge.setPadding(d(9), d(3), d(9), d(3));
        capRow.addView(liveBadge);
        card.addView(capRow);

        captionScroll = new ScrollView(this);
        captionBody = text("转写文字将在这里实时出现。", 16, Ui.MUTED, 0);
        captionBody.setLineSpacing(Ui.sp(this, 5f), 1f);
        captionScroll.addView(captionBody);
        LinearLayout.LayoutParams bodyLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, d(150));
        bodyLp.topMargin = d(12);
        card.addView(captionScroll, bodyLp);

        // ——— 性能指标 ———
        statsText = mono("RTF --  ·  首字 --  ·  线程 4  ·  PSS --", 11.5f, Ui.MUTED);
        statsText.setLineSpacing(Ui.sp(this, 4f), 1f);
        LinearLayout.LayoutParams statLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        statLp.topMargin = d(14);
        col.addView(statsText, statLp);

        // ——— 操作区 ———
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams actLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        actLp.topMargin = d(20);
        col.addView(actions, actLp);

        benchBtn = text("跑基准", 14.5f, Ui.FG, 0);
        benchBtn.setGravity(Gravity.CENTER);
        benchBtn.setPadding(d(16), d(11), d(16), d(11));
        benchBtn.setBackground(Ui.pressableStroke(this, 0x00000000, Ui.FG_SOFT,
                Ui.R, Ui.BORDER, 1f));
        benchBtn.setEnabled(false);
        benchBtn.setOnClickListener(v -> toggleBench());
        actions.addView(benchBtn, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView clearBtn = text("清空", 14.5f, Ui.MUTED, 0);
        clearBtn.setGravity(Gravity.CENTER);
        clearBtn.setPadding(d(16), d(11), d(16), d(11));
        clearBtn.setOnClickListener(v -> {
            if (!running) {
                committed.clear();
                showCaption("", "");
                updateIdleUi();
            }
        });
        LinearLayout.LayoutParams clearLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        clearLp.leftMargin = d(8);
        actions.addView(clearBtn, clearLp);

        return scroll;
    }

    // ---------------- 录音态 / 空闲态 ----------------

    private void updateIdleUi() {
        recordBtn.setImageResource(R.drawable.ic_mic);
        recordBtn.setBackground(Ui.pressable(this, Ui.ACCENT, Ui.ACCENT_PRESSED, 999f));
        statusDot.setBackground(Ui.oval(this, Ui.BORDER));
        statusDot.setAlpha(1f);
        statusText.setText("点击麦克风开始识别");
        statusText.setTextColor(Ui.MUTED);
        liveBadge.setText("待机");
        liveBadge.setTextColor(Ui.MUTED);
        liveBadge.setBackground(Ui.round(this, Ui.FG_SOFT, 999f));
        waveform.setActive(false);
        stopAnim(ringAnim);
        stopAnim(dotAnim);
        ring.setAlpha(0f);
        ring.setScaleX(1f);
        ring.setScaleY(1f);
    }

    private void updateRecordingUi() {
        recordBtn.setImageResource(R.drawable.ic_stop);
        recordBtn.setBackground(Ui.pressable(this, Ui.DANGER, Ui.DANGER_PRESSED, 999f));
        statusDot.setBackground(Ui.oval(this, Ui.DANGER));
        statusText.setText("正在聆听…");
        statusText.setTextColor(Ui.FG);
        liveBadge.setText("LIVE");
        liveBadge.setTextColor(Ui.ACCENT);
        liveBadge.setBackground(Ui.round(this, Ui.ACCENT_SOFT, 999f));
        waveform.setActive(true);

        ringAnim = ObjectAnimator.ofFloat(ring, "scaleX", 0.82f, 1.16f);
        ObjectAnimator sy = ObjectAnimator.ofFloat(ring, "scaleY", 0.82f, 1.16f);
        ObjectAnimator al = ObjectAnimator.ofFloat(ring, "alpha", 0.8f, 0f);
        android.animation.AnimatorSet set = new android.animation.AnimatorSet();
        set.playTogether(ringAnim, sy, al);
        set.setDuration(1600);
        set.setInterpolator(new android.view.animation.LinearInterpolator());
        set.start();
        ringAnim = set;

        dotAnim = ObjectAnimator.ofFloat(statusDot, "alpha", 1f, 0.25f);
        dotAnim.setDuration(1100);
        dotAnim.setRepeatCount(ValueAnimator.INFINITE);
        dotAnim.setRepeatMode(ValueAnimator.REVERSE);
        dotAnim.start();
    }

    private void stopAnim(android.animation.Animator a) {
        if (a != null) a.cancel();
    }

    // ---------------- 模型 ----------------

    private File externalModelDir() {
        String sub = getIntent().getStringExtra("modelDir");
        return new File(getExternalFilesDir(null), sub == null ? "model" : sub);
    }

    private String readLabel(File dir) {
        try {
            File lf = new File(dir, "LABEL.txt");
            if (lf.exists()) {
                String s = new String(java.nio.file.Files.readAllBytes(lf.toPath()),
                        "UTF-8").trim();
                if (!s.isEmpty()) return s;
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private void copyAsset(String name, TextView progressHook) throws java.io.IOException {
        File out = new File(getFilesDir(), "model/" + name);
        if (out.exists() && out.length() > 0) return;
        out.getParentFile().mkdirs();
        long total = 0;
        try (java.io.InputStream in = getAssets().open("model/" + name);
             java.io.FileOutputStream fos = new java.io.FileOutputStream(out)) {
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) > 0) {
                fos.write(buf, 0, n);
                total += n;
                if (progressHook != null && total % (16 << 20) < (1 << 20)) {
                    final String msg = String.format("首次解包 %s %.0fMB",
                            name, total / 1048576.0);
                    runOnUiThread(() -> progressHook.setText(msg));
                }
            }
        }
    }

    /** 外部目录优先（A/B 用），否则解包内嵌模型。返回 {enc, dec, joiner, tokens}。 */
    private String[] resolveModelFiles() throws java.io.IOException {
        File ext = externalModelDir();
        File[] list = ext.listFiles((dir, n) -> n.endsWith(".onnx") || n.equals("tokens.txt"));
        if (list != null) {
            String enc = null, dec = null, join = null, tok = null;
            for (File f : list) {
                String n = f.getName();
                if (n.startsWith("encoder")) enc = f.getAbsolutePath();
                else if (n.startsWith("decoder")) dec = f.getAbsolutePath();
                else if (n.startsWith("joiner")) join = f.getAbsolutePath();
                else if (n.equals("tokens.txt")) tok = f.getAbsolutePath();
            }
            if (enc != null && dec != null && join != null && tok != null) {
                String l = readLabel(ext);
                modelLabel = l != null ? l : ext.getName();
                return new String[]{enc, dec, join, tok};
            }
        }

        String[] names = {"encoder-epoch-99-avg-1.int8.onnx",
                "decoder-epoch-99-avg-1.onnx",
                "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"};

        String assetLabel = "内嵌模型";
        try (java.io.InputStream in = getAssets().open("model/LABEL.txt")) {
            byte[] b = new byte[256];
            int n = in.read(b);
            if (n > 0) assetLabel = new String(b, 0, n, "UTF-8").trim();
        } catch (Exception ignored) {
        }

        // 用 LABEL 作为版本戳：换模型后解包目录必须整体替换，否则会混用不同模型的权重
        File dir = new File(getFilesDir(), "model");
        File stamp = new File(dir, ".stamp");
        String old = "";
        try {
            if (stamp.exists()) {
                old = new String(java.nio.file.Files.readAllBytes(stamp.toPath()), "UTF-8").trim();
            }
        } catch (Exception ignored) {
        }
        if (!old.equals(assetLabel)) {
            if (dir.exists()) {
                File[] olds = dir.listFiles();
                if (olds != null) for (File f : olds) f.delete();
            }
            dir.mkdirs();
        }
        modelLabel = assetLabel;
        for (String n : names) copyAsset(n, null);
        try (java.io.FileOutputStream fos = new java.io.FileOutputStream(stamp)) {
            fos.write(assetLabel.getBytes("UTF-8"));
        } catch (Exception ignored) {
        }
        String base = dir.getAbsolutePath() + "/";
        return new String[]{base + names[0], base + names[1], base + names[2], base + names[3]};
    }

    private void loadModel() {
        try {
            long t0 = SystemClock.elapsedRealtime();
            String[] f = resolveModelFiles();

            OnlineTransducerModelConfig trans = OnlineTransducerModelConfig.builder()
                    .setEncoder(f[0]).setDecoder(f[1]).setJoiner(f[2]).build();
            OnlineModelConfig mc = OnlineModelConfig.builder()
                    .setTransducer(trans).setTokens(f[3])
                    .setNumThreads(numThreads).setProvider(provider).setDebug(false).build();
            FeatureConfig fc = FeatureConfig.builder()
                    .setSampleRate(SAMPLE_RATE).setFeatureDim(80).build();
            EndpointConfig ec = EndpointConfig.builder()
                    .setRule1(EndpointRule.builder().setMustContainNonSilence(false)
                            .setMinTrailingSilence(2.4f).setMinUtteranceLength(0f).build())
                    .setRule2(EndpointRule.builder().setMustContainNonSilence(true)
                            .setMinTrailingSilence(0.6f).setMinUtteranceLength(0f).build())
                    .setRule3(EndpointRule.builder().setMustContainNonSilence(false)
                            .setMinTrailingSilence(0f).setMinUtteranceLength(15f).build())
                    .build();
            OnlineRecognizerConfig cfg = OnlineRecognizerConfig.builder()
                    .setFeatureConfig(fc).setOnlineModelConfig(mc).setEndpointConfig(ec)
                    .setDecodingMethod("greedy_search").setEnableEndpoint(true).build();

            recognizer = new OnlineRecognizer(cfg);
            long ms = SystemClock.elapsedRealtime() - t0;
            runOnUiThread(() -> {
                modelPill.setText(modelLabel);
                statusText.setText(String.format("模型就绪 · 加载 %.1fs", ms / 1000f));
                benchBtn.setEnabled(true);
                if (getIntent().getBooleanExtra("autoBench", false)) {
                    benchBtn.postDelayed(this::toggleBench, 400);
                }
            });
        } catch (Exception e) {
            runOnUiThread(() -> statusText.setText("模型加载失败：" + e.getMessage()));
        }
    }

    // ---------------- 麦克风实时字幕 ----------------

    private void toggleMic() {
        if (benchRunning) return;
        if (running) {
            running = false;
            return;
        }
        committed.clear();
        showCaption("", "");
        timerText.setText("00:00");
        running = true;
        updateRecordingUi();
        new Thread(this::recordLoop).start();
    }

    private void recordLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        int chunk = SAMPLE_RATE * chunkMs / 1000;
        int minBuf = AudioRecord_minBuffer(chunk);
        android.media.AudioRecord ar = new android.media.AudioRecord(
                android.media.MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE, android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_FLOAT, Math.max(minBuf, chunk * 16));
        ar.startRecording();

        OnlineStream stream = recognizer.createStream();
        float[] buf = new float[chunk];
        long startMs = SystemClock.elapsedRealtime();
        long firstTextMs = -1;
        double audioSec = 0, decodeSec = 0, lagMs = 0;
        List<Double> chunkMsList = new ArrayList<>();
        String partial = "";
        int iter = 0;

        while (running) {
            int n = ar.read(buf, 0, chunk, android.media.AudioRecord.READ_BLOCKING);
            if (n <= 0) continue;
            float[] samples = (n == chunk) ? buf : Arrays.copyOf(buf, n);
            lastRms = rmsOf(samples);
            stream.acceptWaveform(samples, SAMPLE_RATE);

            long t0 = SystemClock.elapsedRealtimeNanos();
            while (recognizer.isReady(stream)) recognizer.decode(stream);
            double dt = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9;
            decodeSec += dt;
            chunkMsList.add(dt * 1000);
            audioSec += n / (double) SAMPLE_RATE;
            lagMs = Math.max(0, lagMs + dt * 1000 - n * 1000.0 / SAMPLE_RATE);

            String textNow = recognizer.getResult(stream).getText().trim();
            if (!textNow.isEmpty() && firstTextMs < 0) {
                firstTextMs = SystemClock.elapsedRealtime() - startMs;
            }
            boolean endpoint = recognizer.isEndpoint(stream);
            if (endpoint) {
                if (!textNow.isEmpty()) committed.add(textNow);
                recognizer.reset(stream);
                partial = "";
            } else {
                partial = textNow;
            }

            iter++;
            if (endpoint || iter % UI_CAPTION_EVERY == 0) {
                String p = partial;
                runOnUiThread(() -> {
                    showCaption(TextUtils.join("\n", committed), p);
                    waveform.pushLevel(lastRms);
                });
            }
            if (iter % UI_STATS_EVERY == 0) {
                final double a = audioSec, dec = decodeSec, lag = lagMs;
                final long fst = firstTextMs;
                final List<Double> chunks = new ArrayList<>(chunkMsList);
                runOnUiThread(() -> updateStats(a, dec, chunks, fst, lag, true));
            }
        }

        stream.inputFinished();
        while (recognizer.isReady(stream)) recognizer.decode(stream);
        String tail = recognizer.getResult(stream).getText().trim();
        if (!tail.isEmpty()) committed.add(tail);
        stream.release();
        ar.stop();
        ar.release();

        final double aSec = audioSec, dSec = decodeSec, lagFinal = lagMs;
        final long fFirst = firstTextMs;
        final List<Double> chunksFinal = new ArrayList<>(chunkMsList);
        runOnUiThread(() -> {
            showCaption(TextUtils.join("\n", committed), "");
            updateStats(aSec, dSec, chunksFinal, fFirst, lagFinal, true);
            updateIdleUi();
        });
    }

    private int AudioRecord_minBuffer(int chunk) {
        return android.media.AudioRecord.getMinBufferSize(SAMPLE_RATE,
                android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_FLOAT);
    }

    private static float rmsOf(float[] x) {
        double s = 0;
        for (float v : x) s += v * v;
        return (float) Math.sqrt(s / Math.max(1, x.length));
    }

    private void showCaption(String solid, String partialText) {
        if (solid.isEmpty() && partialText.isEmpty()) {
            captionBody.setText("转写文字将在这里实时出现。");
            captionBody.setTextColor(Ui.MUTED);
            return;
        }
        String head = solid.isEmpty() ? "" : solid + "\n";
        String tailVisible = partialText.isEmpty() ? "" : partialText + " ▌";
        String full = head + tailVisible;
        SpannableString span = new SpannableString(full);
        if (!head.isEmpty()) {
            span.setSpan(new ForegroundColorSpan(Ui.FG), 0, head.length(),
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        if (!tailVisible.isEmpty()) {
            span.setSpan(new ForegroundColorSpan(Ui.MUTED), head.length(), head.length()
                    + (partialText.isEmpty() ? 0 : partialText.length()),
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            span.setSpan(new ForegroundColorSpan(Ui.ACCENT), head.length()
                            + (partialText.isEmpty() ? 0 : partialText.length()),
                    full.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        captionBody.setTextColor(Ui.FG);
        captionBody.setText(span);
        captionScroll.post(() -> captionScroll.fullScroll(View.FOCUS_DOWN));
    }

    private void updateStats(double audioSec, double decodeSec, List<Double> chunkMsList,
                             long firstTextMs, double lagMs, boolean showTimer) {
        double rtf = audioSec > 0 ? decodeSec / audioSec : 0;
        List<Double> sorted = new ArrayList<>(chunkMsList);
        Collections.sort(sorted);
        double p95 = sorted.isEmpty() ? 0
                : sorted.get(Math.min(sorted.size() - 1, (int) (sorted.size() * 0.95)));
        statsText.setText(String.format(
                "RTF %.3f · 解码 p95 %.0fms · 积压 %.0fms\n首字 %s · 音频 %.1fs · %d线程 · PSS %dMB",
                rtf, p95, lagMs,
                firstTextMs < 0 ? "--" : (firstTextMs + "ms"),
                audioSec, numThreads, getPssMb()));
        if (showTimer) {
            int total = (int) audioSec;
            timerText.setText(String.format("%02d:%02d", total / 60, total % 60));
        }
    }

    // ---------------- 文件基准 ----------------

    private void toggleBench() {
        if (benchRunning) {
            benchRunning = false;
            return;
        }
        benchRunning = true;
        benchBtn.setText("停止基准");
        committed.clear();
        showCaption("", "");
        new Thread(this::benchLoop).start();
    }

    private static float[] loadWav16k(File f) throws Exception {
        byte[] raw = java.nio.file.Files.readAllBytes(f.toPath());
        java.nio.ByteBuffer b = java.nio.ByteBuffer.wrap(raw)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.position(12);
        int dataOff = -1, dataLen = 0, sr = 0, ch = 1;
        short fmt = 1, bits = 16;
        while (b.remaining() >= 8) {
            int id = b.getInt();
            int sz = b.getInt();
            if (id == 0x20746d66) {
                fmt = b.getShort(); ch = b.getShort(); sr = b.getInt();
                b.position(b.position() + 6);
                bits = b.getShort();
                b.position(b.position() + sz - 16);
            } else if (id == 0x61746164) {
                dataOff = b.position(); dataLen = sz; break;
            } else {
                b.position(b.position() + sz + (sz & 1));
            }
        }
        if (dataOff < 0 || fmt != 1 || bits != 16) throw new Exception("需16bit PCM wav");
        int n = dataLen / 2;
        float[] mono = new float[n / ch];
        java.nio.ByteBuffer d = java.nio.ByteBuffer.wrap(raw, dataOff, dataLen)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < mono.length; i++) {
            float v = d.getShort() / 32768f;
            for (int c = 1; c < ch; c++) v += d.getShort() / 32768f;
            mono[i] = v / ch;
        }
        if (sr == SAMPLE_RATE) return mono;
        float[] out = new float[(int) ((long) mono.length * SAMPLE_RATE / sr)];
        for (int i = 0; i < out.length; i++) out[i] = mono[(int) ((long) i * sr / SAMPLE_RATE)];
        return out;
    }

    private void benchLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        File dir = getExternalFilesDir("bench");
        File[] wavs = dir != null ? dir.listFiles((d, n) -> n.toLowerCase().endsWith(".wav")) : null;
        if (wavs == null || wavs.length == 0) {
            runOnUiThread(() -> {
                statusText.setText("未找到 wav：" + (dir == null ? "?" : dir.getAbsolutePath()));
                benchRunning = false;
                benchBtn.setText("跑基准");
            });
            return;
        }
        Arrays.sort(wavs);
        int chunk = SAMPLE_RATE * chunkMs / 1000;
        double sumAudio = 0, sumDecode = 0, lagMs = 0;
        List<Long> firsts = new ArrayList<>();
        List<Double> allChunk = new ArrayList<>();
        List<String> lines = new ArrayList<>();

        runOnUiThread(() -> {
            statusText.setText("基准运行中…");
            liveBadge.setText("LIVE");
            liveBadge.setTextColor(Ui.ACCENT);
            liveBadge.setBackground(Ui.round(this, Ui.ACCENT_SOFT, 999f));
            waveform.setActive(true);
        });

        for (File wf : wavs) {
            if (!benchRunning) break;
            float[] samples;
            try {
                samples = loadWav16k(wf);
            } catch (Exception e) {
                lines.add("SKIP " + wf.getName() + "：" + e.getMessage());
                continue;
            }
            OnlineStream stream = recognizer.createStream();
            long startMs = SystemClock.elapsedRealtime();
            long firstMs = -1;
            double audioSec = 0, decodeSec = 0;
            String text = "";
            List<Double> chunkMsList = new ArrayList<>();

            for (int pos = 0; pos < samples.length && benchRunning; pos += chunk) {
                int len = Math.min(chunk, samples.length - pos);
                float[] blk = Arrays.copyOfRange(samples, pos, pos + len);
                lastRms = rmsOf(blk);
                stream.acceptWaveform(blk, SAMPLE_RATE);
                long t0 = SystemClock.elapsedRealtimeNanos();
                while (recognizer.isReady(stream)) recognizer.decode(stream);
                double dt = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9;
                decodeSec += dt;
                chunkMsList.add(dt * 1000);
                allChunk.add(dt * 1000);
                audioSec += len / (double) SAMPLE_RATE;

                String cur = recognizer.getResult(stream).getText().trim();
                if (!cur.isEmpty() && firstMs < 0) firstMs = SystemClock.elapsedRealtime() - startMs;
                text = cur;
                runOnUiThread(() -> waveform.pushLevel(lastRms));

                long behind = (long) (audioSec * 1000) - (SystemClock.elapsedRealtime() - startMs);
                if (behind > 0) SystemClock.sleep(Math.min(behind, chunkMs));
            }
            stream.inputFinished();
            long t0 = SystemClock.elapsedRealtimeNanos();
            while (recognizer.isReady(stream)) recognizer.decode(stream);
            decodeSec += (SystemClock.elapsedRealtimeNanos() - t0) / 1e9;
            text = recognizer.getResult(stream).getText().trim();
            stream.release();

            sumAudio += audioSec;
            sumDecode += decodeSec;
            if (firstMs >= 0) firsts.add(firstMs);
            lines.add(String.format("%s  %.1fs  RTF %.3f  首字 %s\n%s",
                    wf.getName(), audioSec, decodeSec / audioSec,
                    firstMs < 0 ? "--" : (firstMs + "ms"), text));
            final String shown = TextUtils.join("\n\n", lines);
            final double aSec = sumAudio, dSec = sumDecode;
            final long fst = firstMs;
            final List<Double> chunksSnap = new ArrayList<>(allChunk);
            runOnUiThread(() -> {
                captionBody.setTextColor(Ui.FG);
                captionBody.setText(shown);
                captionScroll.post(() -> captionScroll.fullScroll(View.FOCUS_DOWN));
                updateStats(aSec, dSec, chunksSnap, fst, 0, true);
            });
        }

        Collections.sort(allChunk);
        double p95 = allChunk.isEmpty() ? 0
                : allChunk.get(Math.min(allChunk.size() - 1, (int) (allChunk.size() * 0.95)));
        Collections.sort(firsts);
        long fP50 = firsts.isEmpty() ? -1 : firsts.get(firsts.size() / 2);
        double rtf = sumDecode / Math.max(1e-9, sumAudio);
        writeBenchJson(wavs.length, sumAudio, rtf, p95, fP50, lines);

        final int nOk = firsts.size();
        final double sumA = sumAudio, rtfF = rtf, p95F = p95;
        final long p50 = fP50;
        runOnUiThread(() -> {
            statsText.setText(String.format(
                    "基准 %d 条 · %.1fs · p95 %.0fms\nRTF %.3f · 首字 P50 %s · PSS %dMB",
                    nOk, sumA, p95F, rtfF, p50 < 0 ? "--" : (p50 + "ms"), getPssMb()));
            timerText.setText("00:00");
            statusText.setText("基准完成，点击麦克风继续");
            benchBtn.setText("跑基准");
            benchRunning = false;
            updateIdleUi();
        });
    }

    private void writeBenchJson(int n, double audioSec, double rtf, double p95,
                               long firstP50, List<String> lines) {
        try {
            File out = new File(getExternalFilesDir(null), "bench_result.json");
            StringBuilder sb = new StringBuilder();
            sb.append("{\n  \"model\": \"").append(modelLabel).append("\",\n");
            sb.append("  \"threads\": ").append(numThreads).append(",\n");
            sb.append("  \"chunk_ms\": ").append(chunkMs).append(",\n");
            sb.append("  \"n_wav\": ").append(n).append(",\n");
            sb.append("  \"audio_sec\": ").append(String.format("%.2f", audioSec)).append(",\n");
            sb.append("  \"rtf\": ").append(String.format("%.4f", rtf)).append(",\n");
            sb.append("  \"chunk_p95_ms\": ").append(String.format("%.1f", p95)).append(",\n");
            sb.append("  \"first_text_p50_ms\": ").append(firstP50).append(",\n");
            sb.append("  \"pss_mb\": ").append(getPssMb()).append(",\n");
            sb.append("  \"utts\": [\n");
            for (int i = 0; i < lines.size(); i++) {
                sb.append("    \"").append(lines.get(i).replace("\\", "\\\\")
                        .replace("\n", " / ").replace("\"", "\\\"")).append("\"")
                        .append(i + 1 < lines.size() ? "," : "").append("\n");
            }
            sb.append("  ]\n}\n");
            java.io.FileOutputStream fos = new java.io.FileOutputStream(out);
            fos.write(sb.toString().getBytes("UTF-8"));
            fos.close();
        } catch (Exception ignored) {
        }
    }

    private int getPssMb() {
        ActivityManager am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        android.os.Debug.MemoryInfo[] info =
                am.getProcessMemoryInfo(new int[]{Process.myPid()});
        return info[0].getTotalPss() / 1024;
    }

    @Override
    protected void onDestroy() {
        running = false;
        benchRunning = false;
        super.onDestroy();
    }
}
