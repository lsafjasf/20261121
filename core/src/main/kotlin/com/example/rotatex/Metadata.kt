package com.example.rotatex

/**
 * Sidecar metadata written next to every recording (JSON). It logs the locked target
 * orientation, every mid-recording rotation/display-area change with its preview and
 * encoder crops, and whether the audio stream stayed continuous. The emulator demo
 * and the JVM tests verify the produced video against this file.
 */
data class RecordingMetadata(
    val sensorWidth: Int,
    val sensorHeight: Int,
    val sensorOrientationDeg: Int,
    val targetRotationDeg: Int,
    val encodedMetadataRotationDeg: Int,
    val events: MutableList<Event> = mutableListOf(),
    var audioInterrupted: Boolean = false,
) {
    data class Event(
        val tMs: Long,
        val type: Type,
        val displayRotationDeg: Int,
        val previewCropSensor: List<Double>,
        val encoderCropSensor: List<Double>,
        val cropsMatch: Boolean,
    ) {
        enum class Type { RECORD_START, ROTATION, DISPLAY_AREA_CHANGE, RECORD_STOP }
    }

    /** Invariants the demo and tests assert against the sidecar. */
    fun validate(): List<String> {
        val problems = mutableListOf<String>()
        if (audioInterrupted) problems += "audio stream was interrupted"
        val encoderCrops = events.map { it.encoderCropSensor }.toSet()
        if (encoderCrops.size > 1) problems += "encoder crop changed mid-recording: $encoderCrops"
        events.forEachIndexed { i, e ->
            if (!e.cropsMatch) problems += "event[$i] ${e.type}: preview/encoder crop mismatch"
        }
        val expectedMeta = TargetTransform.encodedMetadataRotation(sensorOrientationDeg, targetRotationDeg)
        if (encodedMetadataRotationDeg != expectedMeta) {
            problems += "encoded rotation metadata $encodedMetadataRotationDeg != expected $expectedMeta"
        }
        return problems
    }

    fun toJson(): String {
        val sb = StringBuilder()
        sb.append("{\n")
        sb.append("  \"sensorSize\": [").append(sensorWidth).append(", ").append(sensorHeight).append("],\n")
        sb.append("  \"sensorOrientationDeg\": ").append(sensorOrientationDeg).append(",\n")
        sb.append("  \"targetRotationDeg\": ").append(targetRotationDeg).append(",\n")
        sb.append("  \"encodedMetadataRotationDeg\": ").append(encodedMetadataRotationDeg).append(",\n")
        sb.append("  \"audioInterrupted\": ").append(audioInterrupted).append(",\n")
        sb.append("  \"events\": [\n")
        events.forEachIndexed { i, e ->
            sb.append("    {")
            sb.append("\"tMs\": ").append(e.tMs)
            sb.append(", \"type\": \"").append(e.type.name).append('"')
            sb.append(", \"displayRotationDeg\": ").append(e.displayRotationDeg)
            sb.append(", \"previewCropSensor\": ").append(e.previewCropSensor.toJsonArray())
            sb.append(", \"encoderCropSensor\": ").append(e.encoderCropSensor.toJsonArray())
            sb.append(", \"cropsMatch\": ").append(e.cropsMatch)
            sb.append(if (i == events.lastIndex) "}\n" else "},\n")
        }
        sb.append("  ]\n")
        sb.append("}\n")
        return sb.toString()
    }

    private fun List<Double>.toJsonArray(): String =
        joinToString(", ", "[", "]") { "%.1f".format(it) }
}
