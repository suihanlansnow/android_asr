package com.example.asrdemo;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/** 波形条：空闲时静止 14% 高，录音时按最近音量滚动并带呼吸起伏。 */
class WaveformView extends View {
    private static final int BARS = 27;
    private final float[] levels = new float[BARS];
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF bar = new RectF();
    private boolean active = false;
    private long phase = 0;

    WaveformView(Context c, AttributeSet attrs) {
        super(c, attrs);
        paint.setStyle(Paint.Style.FILL);
    }

    void setActive(boolean active) {
        this.active = active;
        invalidate();
    }

    /** rms 为 0..1 的归一化音量。 */
    void pushLevel(float rms) {
        System.arraycopy(levels, 1, levels, 0, BARS - 1);
        float v = Math.min(1f, rms * 6f);
        levels[BARS - 1] = v;
        phase++;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        float w = getWidth(), h = getHeight();
        if (w <= 0 || h <= 0) return;
        float barW = Ui.dp(getContext(), 3);
        float gap = Ui.dp(getContext(), 3);
        float total = BARS * barW + (BARS - 1) * gap;
        float x = (w - total) / 2f;
        float mid = h / 2f;
        paint.setColor(active ? 0x572E6BE6 : Ui.BORDER);

        for (int i = 0; i < BARS; i++) {
            float frac = active ? Math.max(0.14f, levels[i]) : 0.14f;
            if (active) {
                // 呼吸：越靠近中间越高，形成录音中的流动感
                float wob = (float) (Math.sin((phase * 0.55f) + i * 0.42f) * 0.14f);
                frac = Math.max(0.12f, Math.min(1f, frac + wob));
            }
            float bh = h * frac;
            bar.set(x, mid - bh / 2f, x + barW, mid + bh / 2f);
            canvas.drawRoundRect(bar, barW / 2f, barW / 2f, paint);
            x += barW + gap;
        }
    }
}
