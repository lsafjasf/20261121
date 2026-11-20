package com.example.rotatex.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View

/**
 * Debug HUD overlay ("预览裁切和编码裁切核对"): draws the preview crop window and the
 * encoder crop window in normalized sensor coordinates, plus a text block with both
 * rects and the MATCH/MISMATCH verdict from [com.example.rotatex.CropReconciler].
 */
class CropHudView(context: Context) : View(context) {

    private var state: RecordingEngine.HudState? = null

    private val previewPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val encoderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(12f, 8f), 0f)
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }
    private val bgPaint = Paint().apply {
        color = 0x88000000.toInt()
    }

    fun update(state: RecordingEngine.HudState) {
        this.state = state
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val s = state ?: return
        val sensorW = 1280f
        val sensorH = 720f
        // Normalized sensor coordinates mapped onto a small HUD region (top-right).
        val side = width * 0.35f
        val scale = side / maxOf(sensorW, sensorH)
        val originX = width - side - 16f
        val originY = 16f
        fun toView(r: com.example.rotatex.RectD) = RectF(
            originX + (r.left * scale).toFloat(),
            originY + (r.top * scale).toFloat(),
            originX + (r.right * scale).toFloat(),
            originY + (r.bottom * scale).toFloat(),
        )
        s.previewCropSensor?.let { canvas.drawRect(toView(it), previewPaint) }
        s.encoderCropSensor?.let { canvas.drawRect(toView(it), encoderPaint) }

        val verdict = when {
            s.report == null -> "N/A"
            s.report.matches -> "MATCH"
            else -> "MISMATCH"
        }
        val lines = buildList {
            add("旋转 ${s.displayRotationDeg}°  成片方向 ${s.targetRotationDeg}°")
            add("录制 ${if (s.recording) "REC" else "停止"}  音频 ${if (s.audioActive) "连续" else "—"}")
            add("展示区 ${s.displayArea}")
            add("核对 $verdict")
            s.previewCropSensor?.let { add("预览裁切(sensor) $it") }
            s.encoderCropSensor?.let { add("编码裁切(sensor) $it") }
        }
        val pad = 16f
        val lineHeight = textPaint.textSize + 12f
        val boxHeight = pad * 2 + lineHeight * lines.size
        canvas.drawRect(0f, height - boxHeight, width.toFloat(), height.toFloat(), bgPaint)
        lines.forEachIndexed { i, line ->
            canvas.drawText(line, pad, height - boxHeight + pad + lineHeight * (i + 0.75f), textPaint)
        }
    }
}
