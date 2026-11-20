package com.example.rotatex

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** Sidecar metadata invariants: constant encoder crop, matching crops, continuous audio. */
class MetadataTest {

    private fun event(tMs: Long, type: RecordingMetadata.Event.Type, display: Int, match: Boolean = true) =
        RecordingMetadata.Event(
            tMs = tMs,
            type = type,
            displayRotationDeg = display,
            previewCropSensor = listOf(0.0, 45.0, 1280.0, 675.0),
            encoderCropSensor = listOf(0.0, 45.0, 1280.0, 675.0),
            cropsMatch = match,
        )

    @Test
    fun `clean recording across four rotations validates`() {
        val meta = RecordingMetadata(
            sensorWidth = 1280,
            sensorHeight = 720,
            sensorOrientationDeg = 90,
            targetRotationDeg = 0,
            encodedMetadataRotationDeg = 90,
        )
        meta.events += event(0, RecordingMetadata.Event.Type.RECORD_START, 0)
        meta.events += event(1_000, RecordingMetadata.Event.Type.ROTATION, 90)
        meta.events += event(2_000, RecordingMetadata.Event.Type.ROTATION, 180)
        meta.events += event(3_000, RecordingMetadata.Event.Type.ROTATION, 270)
        meta.events += event(4_000, RecordingMetadata.Event.Type.RECORD_STOP, 0)
        assertEquals(emptyList<String>(), meta.validate())
    }

    @Test
    fun `audio interruption is reported`() {
        val meta = RecordingMetadata(1280, 720, 90, 0, 90, audioInterrupted = true)
        assertTrue(meta.validate().any { it.contains("audio") })
    }

    @Test
    fun `crop mismatch event is reported`() {
        val meta = RecordingMetadata(1280, 720, 90, 0, 90)
        meta.events += event(0, RecordingMetadata.Event.Type.RECORD_START, 0, match = false)
        assertTrue(meta.validate().any { it.contains("mismatch") })
    }

    @Test
    fun `wrong encoded rotation metadata is reported`() {
        val meta = RecordingMetadata(1280, 720, 90, 0, encodedMetadataRotationDeg = 0)
        assertTrue(meta.validate().any { it.contains("metadata") })
    }

    @Test
    fun `json contains the key fields`() {
        val meta = RecordingMetadata(1280, 720, 90, 0, 90)
        meta.events += event(0, RecordingMetadata.Event.Type.RECORD_START, 0)
        val json = meta.toJson()
        assertTrue(json.contains("\"sensorOrientationDeg\": 90"))
        assertTrue(json.contains("\"targetRotationDeg\": 0"))
        assertTrue(json.contains("\"encodedMetadataRotationDeg\": 90"))
        assertTrue(json.contains("\"audioInterrupted\": false"))
        assertTrue(json.contains("RECORD_START"))
    }
}
