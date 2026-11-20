package com.example.rotatex.sim

import com.example.rotatex.Anchor
import com.example.rotatex.CropReconciler
import com.example.rotatex.RectD
import com.example.rotatex.RecordingMetadata
import com.example.rotatex.Size
import com.example.rotatex.TargetTransform
import java.awt.image.BufferedImage

/**
 * JVM simulation of the CameraX pipeline used by the app:
 *
 *  sensor buffer -> rotate CW by relativeRotation -> upright buffer -> ViewPort crop
 *
 * The encoder path locks its rotation to the selected target orientation at record
 * start; the preview path re-rotates and re-crops on every display rotation. Both
 * crops are reconciled in sensor coordinates, exactly like the app's debug HUD.
 */
class Simulator(
    val sensor: Size = Size(1280, 720),
    val sensorOrientationDeg: Int = 90,
    val targetRotationDeg: Int = 0,
    val aspectW: Int = 16,
    val aspectH: Int = 9,
    val anchor: Anchor = Anchor.CENTER,
    val worldArrowDeg: Int = 0,
) {
    data class Frame(
        val displayRotationDeg: Int,
        val preview: BufferedImage,
        val encoded: BufferedImage,
        val playback: BufferedImage,
        val previewCropSensor: RectD,
        val encoderCropSensor: RectD,
        val cropsMatch: Boolean,
    )

    private val encoderRel = TargetTransform.relativeRotation(sensorOrientationDeg, targetRotationDeg)
    private val encoderUprightCrop: RectD
    private val encoderSensorCrop: RectD

    init {
        val (upright, sensorCrop) = CropReconciler.encoderWindow(
            sensor, sensorOrientationDeg, targetRotationDeg, aspectW, aspectH, anchor,
        )
        encoderUprightCrop = upright
        encoderSensorCrop = sensorCrop
    }

    val metadata = RecordingMetadata(
        sensorWidth = sensor.width,
        sensorHeight = sensor.height,
        sensorOrientationDeg = sensorOrientationDeg,
        targetRotationDeg = targetRotationDeg,
        encodedMetadataRotationDeg = TargetTransform.encodedMetadataRotation(sensorOrientationDeg, targetRotationDeg),
    )

    /** Produces the preview/encoded/playback frames for one display rotation. */
    fun frameAt(displayRotationDeg: Int, tMs: Long, type: RecordingMetadata.Event.Type): Frame {
        val arrowInBuffer = TargetTransform.arrowInSensorBuffer(worldArrowDeg, displayRotationDeg, sensorOrientationDeg)
        val sensorFrame = ArrowTarget.render(sensor, arrowInBuffer)
        // Preview path: rotate for the display, then show exactly the encoder window.
        val previewRel = TargetTransform.relativeRotation(sensorOrientationDeg, displayRotationDeg)
        val previewUprightCrop = CropReconciler.previewCropMatchingEncoder(
            sensor, sensorOrientationDeg, displayRotationDeg, encoderSensorCrop,
        )
        val preview = crop(rotateClockwise(sensorFrame, previewRel), previewUprightCrop)
        // Encoder path: locked at record start; the file stores the cropped sensor
        // window plus the rotation metadata, players rotate on playback.
        val encoded = crop(sensorFrame, encoderSensorCrop)
        val playback = rotateClockwise(encoded, encoderRel)
        val report = CropReconciler.verify(
            previewUprightCrop, previewRel, encoderUprightCrop, encoderRel, sensor,
        )
        metadata.events += RecordingMetadata.Event(
            tMs = tMs,
            type = type,
            displayRotationDeg = displayRotationDeg,
            previewCropSensor = report.previewSensorRect.toList(),
            encoderCropSensor = report.encoderSensorRect.toList(),
            cropsMatch = report.matches,
        )
        return Frame(
            displayRotationDeg = displayRotationDeg,
            preview = preview,
            encoded = encoded,
            playback = playback,
            previewCropSensor = report.previewSensorRect,
            encoderCropSensor = report.encoderSensorRect,
            cropsMatch = report.matches,
        )
    }

    companion object {
        fun rotateClockwise(src: BufferedImage, deg: Int): BufferedImage {
            val d = Math.floorMod(deg, 360)
            if (d == 0) return src
            val w = src.width
            val h = src.height
            val (outW, outH) = if (d == 180) w to h else h to w
            val out = BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val (nx, ny) = when (d) {
                        90 -> (h - 1 - y) to x
                        180 -> (w - 1 - x) to (h - 1 - y)
                        else -> y to (w - 1 - x) // 270
                    }
                    out.setRGB(nx, ny, src.getRGB(x, y))
                }
            }
            return out
        }

        fun crop(src: BufferedImage, rect: RectD): BufferedImage {
            val l = rect.left.toInt().coerceIn(0, src.width - 1)
            val t = rect.top.toInt().coerceIn(0, src.height - 1)
            val w = rect.width.toInt().coerceAtLeast(1).coerceAtMost(src.width - l)
            val h = rect.height.toInt().coerceAtLeast(1).coerceAtMost(src.height - t)
            return src.getSubimage(l, t, w, h)
        }
    }
}
