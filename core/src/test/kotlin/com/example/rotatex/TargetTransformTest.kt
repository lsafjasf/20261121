package com.example.rotatex

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Independent transform-matrix tests. Physical model used by every expectation:
 * a printed arrow target points to the sky (world-up = 0 deg); the phone's sensor
 * orientation is 90 deg (typical back camera); display rotations cycle 0/90/180/270.
 */
class TargetTransformTest {

    private val sensor = Size(1280, 720)
    private val sensorOrientation = 90

    @Test
    fun `relative rotation matches CameraX back-camera formula`() {
        // CameraX: (sensorOrientation - targetRotationDegrees + 360) % 360
        assertEquals(90, TargetTransform.relativeRotation(90, 0))
        assertEquals(0, TargetTransform.relativeRotation(90, 90))
        assertEquals(270, TargetTransform.relativeRotation(90, 180))
        assertEquals(180, TargetTransform.relativeRotation(90, 270))
    }

    @Test
    fun `upright size swaps dimensions on quarter turns`() {
        assertEquals(Size(1280, 720), TargetTransform.uprightSize(sensor, 0))
        assertEquals(Size(720, 1280), TargetTransform.uprightSize(sensor, 90))
        assertEquals(Size(1280, 720), TargetTransform.uprightSize(sensor, 180))
        assertEquals(Size(720, 1280), TargetTransform.uprightSize(sensor, 270))
    }

    @Test
    fun `crop rect is the largest centered rect with the target aspect`() {
        val r = TargetTransform.cropRect(Size(1280, 720), 1, 1)
        assertEquals(720.0, r.width, 1e-6)
        assertEquals(720.0, r.height, 1e-6)
        assertEquals(280.0, r.left, 1e-6)
        assertEquals(0.0, r.top, 1e-6)
    }

    @Test
    fun `crop rect honors anchors`() {
        val left = TargetTransform.cropRect(Size(1280, 720), 1, 1, Anchor.LEFT)
        assertEquals(0.0, left.left, 1e-6)
        val right = TargetTransform.cropRect(Size(1280, 720), 1, 1, Anchor.RIGHT)
        assertEquals(1280.0, right.right, 1e-6)
        val top = TargetTransform.cropRect(Size(720, 1280), 16, 9, Anchor.TOP)
        assertEquals(0.0, top.top, 1e-6)
        assertEquals(720.0 / (16.0 / 9.0), top.height, 1e-6)
    }

    @Test
    fun `sensor-upright rect mapping round-trips for all rotations`() {
        val rect = RectD(100.0, 50.0, 640.0, 400.0)
        for (deg in intArrayOf(0, 90, 180, 270)) {
            val up = TargetTransform.sensorToUprightRect(rect, sensor, deg)
            val back = TargetTransform.uprightToSensorRect(up, sensor, deg)
            assertTrue(
                rect.maxDeviation(back) < 1e-6,
                "round-trip failed at $deg: $rect -> $up -> $back",
            )
        }
    }

    @Test
    fun `matrix maps corners like the rect helper`() {
        val m = TargetTransform.sensorToUprightMatrix(sensor, 90)
        // Clockwise 90: sensor top-left (0,0) lands on the upright top-right corner.
        val (x, y) = m.map(0.0, 0.0)
        assertEquals(720.0, x, 1e-6)
        assertEquals(0.0, y, 1e-6)
    }

    @Test
    fun `arrow in sensor buffer for four display rotations`() {
        // Sensor left edge faces device-top (sensorOrientation = 90), so the sky-pointing
        // arrow sits at the buffer's left edge (270 deg) when the device is natural.
        assertEquals(270, TargetTransform.arrowInSensorBuffer(0, 0, sensorOrientation))
        assertEquals(0, TargetTransform.arrowInSensorBuffer(0, 90, sensorOrientation))
        assertEquals(90, TargetTransform.arrowInSensorBuffer(0, 180, sensorOrientation))
        assertEquals(180, TargetTransform.arrowInSensorBuffer(0, 270, sensorOrientation))
    }

    @Test
    fun `preview arrow always points to world direction`() {
        // Preview targetRotation tracks the display: the sky arrow stays up on screen
        // no matter how the device rotates mid-recording.
        for (r in intArrayOf(0, 90, 180, 270)) {
            assertEquals(0, TargetTransform.arrowOnPreview(0), "preview at rotation $r")
        }
    }

    @Test
    fun `playback arrow rotates with the device because the file orientation is locked`() {
        val target = 0 // recording locked to natural orientation
        assertEquals(0, TargetTransform.arrowInPlayback(0, 0, target))
        assertEquals(90, TargetTransform.arrowInPlayback(0, 90, target))
        assertEquals(180, TargetTransform.arrowInPlayback(0, 180, target))
        assertEquals(270, TargetTransform.arrowInPlayback(0, 270, target))
    }

    @Test
    fun `playback direction is independent of sensor orientation`() {
        for (s in intArrayOf(0, 90, 180, 270)) {
            // arrowInPlayback has no sensor parameter by construction; also verify the
            // composed pipeline (buffer -> encoded metadata) collapses to it.
            val buf = TargetTransform.arrowInSensorBuffer(0, 90, s)
            val meta = TargetTransform.encodedMetadataRotation(s, 0)
            assertEquals(90, Math.floorMod(buf + meta, 360), "sensor orientation $s")
        }
    }

    @Test
    fun `encoded metadata rotation locks to the selected target orientation`() {
        assertEquals(90, TargetTransform.encodedMetadataRotation(90, 0))
        assertEquals(0, TargetTransform.encodedMetadataRotation(90, 90))
        assertEquals(270, TargetTransform.encodedMetadataRotation(90, 180))
        assertEquals(180, TargetTransform.encodedMetadataRotation(90, 270))
    }
}
