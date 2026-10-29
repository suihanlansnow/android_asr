package com.example.asrdemo;

import android.content.Context;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.StateListDrawable;
import android.util.TypedValue;

/** 设计令牌：取色自 res/voice-notepad.html 的 :root 变量（oklch 转近似 sRGB）。 */
final class Ui {
    static final int BG = 0xFFFCFCFD;
    static final int SURFACE = 0xFFFFFFFF;
    static final int FG = 0xFF1A1D24;
    static final int MUTED = 0xFF767C88;
    static final int BORDER = 0xFFE6E8EC;
    static final int ACCENT = 0xFF2E6BE6;
    static final int ACCENT_PRESSED = 0xFF2258C4;
    static final int ACCENT_SOFT = 0x242E6BE6;
    static final int ACCENT_RING = 0x662E6BE6;
    static final int DANGER = 0xFFE0413C;
    static final int DANGER_PRESSED = 0xFFC1332F;
    static final int DANGER_SOFT = 0x1AE0413C;
    static final int SUCCESS = 0xFF3FA46A;
    static final int FG_SOFT = 0x0F1A1D24;

    static final float R_SM = 8f;
    static final float R = 10f;
    static final float R_LG = 14f;

    private Ui() {}

    static int dp(Context c, float v) {
        return (int) (v * c.getResources().getDisplayMetrics().density + 0.5f);
    }

    static float sp(Context c, float v) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v,
                c.getResources().getDisplayMetrics());
    }

    static GradientDrawable round(Context c, int color, float radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.RECTANGLE);
        d.setColor(color);
        d.setCornerRadius(dp(c, radiusDp));
        return d;
    }

    static GradientDrawable roundStroke(Context c, int fill, float radiusDp,
                                        int strokeColor, float strokeDp) {
        GradientDrawable d = round(c, fill, radiusDp);
        d.setStroke(dp(c, strokeDp), strokeColor);
        return d;
    }

    static GradientDrawable oval(Context c, int fill) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(fill);
        return d;
    }

    static GradientDrawable ovalRing(Context c, int strokeColor, float strokeDp) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(0x00000000);
        d.setStroke(dp(c, strokeDp), strokeColor);
        return d;
    }

    /** 按钮：按下反馈由 state_pressed 驱动。 */
    static StateListDrawable pressable(Context c, int normal, int pressed, float radiusDp) {
        StateListDrawable s = new StateListDrawable();
        s.addState(new int[]{android.R.attr.state_pressed}, round(c, pressed, radiusDp));
        s.addState(new int[0], round(c, normal, radiusDp));
        return s;
    }

    static StateListDrawable pressableStroke(Context c, int fill, int fillPressed,
                                             float radiusDp, int strokeColor, float strokeDp) {
        StateListDrawable s = new StateListDrawable();
        s.addState(new int[]{android.R.attr.state_pressed},
                roundStroke(c, fillPressed, radiusDp, strokeColor, strokeDp));
        s.addState(new int[0], roundStroke(c, fill, radiusDp, strokeColor, strokeDp));
        return s;
    }
}
