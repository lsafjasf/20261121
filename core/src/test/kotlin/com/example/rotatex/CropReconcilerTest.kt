package com.example.rotatex

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Preview crop vs encoder crop reconciliation ("预览裁切和编码裁切核对"):
 * with a shared ViewPort aspect, both crops must be the same window in sensor
 * coordinates for every (display rotation x target rotation) combination.
 */
class CropReconcilerTest {

    private val sensor = Size(1280, 720)
    private val sensorOrientation = 90
    private val rotations = intArrayOf(0, 90, 180, 270)

    @Test
    fun `shared crop matches in sensor space for all 16 rotation pairs`() {
        for (display in rotations) {
            for (target in rotations) {
                val report = CropReconciler.verifySharedCrop(
                    sensor, sensorOrientation, display, target, 16, 9,
                )
                assertTrue(report.matches, "display=$display target=$target: $report")
            }
        }
    }

    @Test
    fun `shared crop matches for non-centered anchors and other aspects`() {
        for (display in rotations) {
            for (target in rotations) {
                for (anchor in Anchor.entries) {
                    val report = CropReconciler.verifySharedCrop(
                        sensor, sensorOrientation, display, target, 1, 1, anchor,
                    )
                    assertTrue(report.matches, "display=$display target=$target anchor=$anchor: $report")
                }
            }
        }
    }

    @Test
    fun `mismatch is detected when aspects differ`() {
        val rel = TargetTransform.relativeRotation(sensorOrientation, 0)
        val up = TargetTransform.uprightSize(sensor, rel)
        val previewCrop = TargetTransform.cropRect(up, 16, 9)
        val encoderCrop = TargetTransform.cropRect(up, 1, 1)
        val report = CropReconciler.verify(previewCrop, rel, encoderCrop, rel, sensor)
        assertFalse(report.matches)
        assertTrue(report.maxDeviation > 1.0)
    }

    @Test
    fun `naive same-aspect crops diverge across quarter turns - why we map via sensor space`() {
        // Cropping 16:9 independently in each upright space yields different sensor
        // windows when display and target rotations differ by 90 degrees. This is the
        // bug the reconciler guards against: the preview ViewPort must be derived from
        // the encoder's sensor-space window, not recomputed from the aspect ratio.
        val previewRel = TargetTransform.relativeRotation(sensorOrientation, 0)
        val encoderRel = TargetTransform.relativeRotation(sensorOrientation, 90)
        val previewCrop = TargetTransform.cropRect(TargetTransform.uprightSize(sensor, previewRel), 16, 9)
        val encoderCrop = TargetTransform.cropRect(TargetTransform.uprightSize(sensor, encoderRel), 16, 9)
        val report = CropReconciler.verify(previewCrop, previewRel, encoderCrop, encoderRel, sensor)
        assertFalse(report.matches)
    }

    @Test
    fun `encoder crop in sensor space is stable while the device rotates`() {
        // The encoder ViewPort is locked at record start (target rotation 0); rotating
        // the device must not move the encoded window. This is what keeps the recorded
        // framing steady while the preview re-crops for the new display rotation.
        val encoderRel = TargetTransform.relativeRotation(sensorOrientation, 0)
        val encoderCrop = TargetTransform.cropRect(
            TargetTransform.uprightSize(sensor, encoderRel), 16, 9,
        )
        val encoderSensorRect = TargetTransform.uprightToSensorRect(encoderCrop, sensor, encoderRel)
        // 16:9 portrait output of a landscape sensor: a centered 405x720 sensor window.
        assertEquals(RectD(437.5, 0.0, 842.5, 720.0), encoderSensorRect)
    }
}
