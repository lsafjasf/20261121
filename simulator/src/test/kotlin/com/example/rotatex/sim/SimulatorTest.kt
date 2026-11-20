package com.example.rotatex.sim

import com.example.rotatex.RecordingMetadata
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * End-to-end verification with the arrow test target in all four orientations:
 * preview stays upright, the encoded file keeps the locked target orientation,
 * and the sidecar metadata validates.
 */
class SimulatorTest {

    private val rotations = intArrayOf(0, 90, 180, 270)

    private fun recordAcrossRotations(sim: Simulator) =
        rotations.mapIndexed { i, r ->
            val type = if (i == 0) {
                RecordingMetadata.Event.Type.RECORD_START
            } else {
                RecordingMetadata.Event.Type.ROTATION
            }
            sim.frameAt(r, tMs = i * 1_000L, type = type)
        }

    @Test
    fun `preview arrow points up on screen at every rotation`() {
        val sim = Simulator()
        for (frame in recordAcrossRotations(sim)) {
            assertEquals(
                0,
                ArrowTarget.detectArrowDirection(frame.preview),
                "preview at R=${frame.displayRotationDeg}",
            )
        }
    }

    @Test
    fun `playback arrow rotates with the device - file keeps target orientation`() {
        val sim = Simulator(targetRotationDeg = 0)
        for (frame in recordAcrossRotations(sim)) {
            assertEquals(
                frame.displayRotationDeg,
                ArrowTarget.detectArrowDirection(frame.playback),
                "playback at R=${frame.displayRotationDeg}",
            )
        }
    }

    @Test
    fun `playback arrow stays upright when target orientation tracks each rotation`() {
        for (r in rotations) {
            val sim = Simulator(targetRotationDeg = r)
            val frame = sim.frameAt(r, 0, RecordingMetadata.Event.Type.RECORD_START)
            assertEquals(0, ArrowTarget.detectArrowDirection(frame.playback), "target=R$r")
        }
    }

    @Test
    fun `edge markers land on expected screen edges per rotation`() {
        // Sensor top edge marker (magenta) appears at screen edge (S - R) CW from top.
        val sim = Simulator() // S = 90
        val expected = mapOf(0 to "right", 90 to "top", 180 to "left", 270 to "bottom")
        for (frame in recordAcrossRotations(sim)) {
            val markers = ArrowTarget.detectEdgeMarkers(frame.preview)
            val magentaEdge = markers.entries.first { it.value == ArrowTarget.EDGE_TOP }.key
            assertEquals(expected[frame.displayRotationDeg], magentaEdge)
        }
    }

    @Test
    fun `encoded window is identical across rotations while preview window rotates`() {
        val sim = Simulator()
        val frames = recordAcrossRotations(sim)
        val first = frames.first()
        for (f in frames) {
            assertEquals(first.encoded.width, f.encoded.width)
            assertEquals(first.encoded.height, f.encoded.height)
            assertTrue(f.encoderCropSensor.maxDeviation(first.encoderCropSensor) < 1e-6)
            assertTrue(f.cropsMatch)
        }
        // Preview window follows the display rotation of the locked encoder window:
        // landscape band at R0/R180, portrait band at R90/R270.
        assertTrue(frames[0].preview.width >= frames[0].preview.height)
        assertTrue(frames[1].preview.height >= frames[1].preview.width)
    }

    @Test
    fun `metadata validates after a four-rotation recording`() {
        val sim = Simulator()
        recordAcrossRotations(sim)
        assertEquals(emptyList<String>(), sim.metadata.validate())
        assertEquals(4, sim.metadata.events.size)
        assertEquals(90, sim.metadata.encodedMetadataRotationDeg)
    }

    @Test
    fun `arrow detection reads the four cardinal directions`() {
        for (deg in rotations) {
            val img = ArrowTarget.render(com.example.rotatex.Size(640, 480), deg)
            assertEquals(deg, ArrowTarget.detectArrowDirection(img), "rendered at $deg")
        }
    }
}
