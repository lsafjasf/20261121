package com.example.rotatex

/**
 * Reconciles the preview crop with the encoder crop ("预览裁切和编码裁切核对").
 *
 * Preview and VideoCapture each see the sensor stream through their own ViewPort
 * crop, expressed in their own upright coordinate spaces (which differ whenever the
 * display rotation differs from the recording's locked target rotation). Mapping both
 * rects back into raw sensor coordinates must yield the same window; otherwise the
 * user is framing something the encoder is not recording.
 */
object CropReconciler {

    data class Report(
        val previewSensorRect: RectD,
        val encoderSensorRect: RectD,
        val maxDeviation: Double,
        val tolerance: Double,
    ) {
        val matches: Boolean get() = maxDeviation <= tolerance

        override fun toString(): String = buildString {
            append(if (matches) "MATCH" else "MISMATCH")
            append(" preview(sensor)=").append(previewSensorRect)
            append(" encoder(sensor)=").append(encoderSensorRect)
            append(" maxDev=%.2f".format(maxDeviation))
        }
    }

    fun verify(
        previewUprightCrop: RectD,
        previewRelRotationDeg: Int,
        encoderUprightCrop: RectD,
        encoderRelRotationDeg: Int,
        sensor: Size,
        tolerance: Double = 1.0,
    ): Report {
        val previewSensor = TargetTransform.uprightToSensorRect(previewUprightCrop, sensor, previewRelRotationDeg)
        val encoderSensor = TargetTransform.uprightToSensorRect(encoderUprightCrop, sensor, encoderRelRotationDeg)
        return Report(
            previewSensorRect = previewSensor,
            encoderSensorRect = encoderSensor,
            maxDeviation = previewSensor.maxDeviation(encoderSensor),
            tolerance = tolerance,
        )
    }

    /**
     * The encoder window: crop rect for the given aspect/anchor in the encoder's
     * upright space (locked to [targetRotationDeg] at record start), plus the same
     * window in raw sensor coordinates. The sensor-space rect is what VideoCapture
     * actually encodes for the whole recording, regardless of later device rotation.
     */
    fun encoderWindow(
        sensor: Size,
        sensorOrientationDeg: Int,
        targetRotationDeg: Int,
        aspectW: Int,
        aspectH: Int,
        anchor: Anchor = Anchor.CENTER,
    ): Pair<RectD, RectD> {
        val encoderRel = TargetTransform.relativeRotation(sensorOrientationDeg, targetRotationDeg)
        val uprightCrop = TargetTransform.cropRect(
            TargetTransform.uprightSize(sensor, encoderRel), aspectW, aspectH, anchor,
        )
        val sensorCrop = TargetTransform.uprightToSensorRect(uprightCrop, sensor, encoderRel)
        return uprightCrop to sensorCrop
    }

    /**
     * Preview crop (in the preview's upright space at [displayRotationDeg]) that shows
     * exactly the encoder window. Because the preview upright space rotates with the
     * display, this rect is the encoder's sensor-space window mapped forward; it is
     * NOT the same as cropping with the same aspect ratio independently in each space
     * whenever display and target rotations differ by 90 degrees.
     */
    fun previewCropMatchingEncoder(
        sensor: Size,
        sensorOrientationDeg: Int,
        displayRotationDeg: Int,
        encoderSensorCrop: RectD,
    ): RectD {
        val previewRel = TargetTransform.relativeRotation(sensorOrientationDeg, displayRotationDeg)
        return TargetTransform.sensorToUprightRect(encoderSensorCrop, sensor, previewRel)
    }

    /**
     * Verifies that the preview crop derived via [previewCropMatchingEncoder] reconciles
     * with the encoder window for the given rotation combination.
     */
    fun verifySharedCrop(
        sensor: Size,
        sensorOrientationDeg: Int,
        displayRotationDeg: Int,
        targetRotationDeg: Int,
        aspectW: Int,
        aspectH: Int,
        anchor: Anchor = Anchor.CENTER,
        tolerance: Double = 1.0,
    ): Report {
        val previewRel = TargetTransform.relativeRotation(sensorOrientationDeg, displayRotationDeg)
        val encoderRel = TargetTransform.relativeRotation(sensorOrientationDeg, targetRotationDeg)
        val (encoderUpright, encoderSensorCrop) = encoderWindow(
            sensor, sensorOrientationDeg, targetRotationDeg, aspectW, aspectH, anchor,
        )
        val previewUpright = previewCropMatchingEncoder(sensor, sensorOrientationDeg, displayRotationDeg, encoderSensorCrop)
        return verify(previewUpright, previewRel, encoderUpright, encoderRel, sensor, tolerance)
    }
}
