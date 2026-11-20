package com.example.rotatex

/** Where the crop window is anchored inside the upright buffer. */
enum class Anchor { CENTER, LEFT, RIGHT, TOP, BOTTOM }

/**
 * Rotation/crop math shared by the CameraX app, the JVM simulator and the tests.
 *
 * Conventions (back camera, matching CameraX):
 *  - The sensor buffer is landscape, size [Size.width] x [Size.height].
 *  - `sensorOrientationDeg` is Camera2 SENSOR_ORIENTATION: degrees to rotate the
 *    sensor buffer clockwise so it is upright at display rotation 0.
 *  - Display/target rotations use Surface.ROTATION_* degrees (0/90/180/270).
 *  - Directions are degrees clockwise from the top edge of the buffer/screen.
 */
object TargetTransform {

    /** Degrees to rotate the sensor buffer clockwise so it appears upright. */
    fun relativeRotation(sensorOrientationDeg: Int, rotationDeg: Int, frontCamera: Boolean = false): Int =
        Math.floorMod(
            if (frontCamera) sensorOrientationDeg + rotationDeg else sensorOrientationDeg - rotationDeg,
            360,
        )

    /** Buffer size after rotating [sensor] clockwise by [relRotationDeg]. */
    fun uprightSize(sensor: Size, relRotationDeg: Int): Size =
        if (Math.floorMod(relRotationDeg, 180) == 0) sensor else sensor.rotated()

    /**
     * Largest rect with aspect [aspectW]:[aspectH] that fits inside [upright],
     * positioned according to [anchor]. This is the crop CameraX computes for a
     * ViewPort with the same aspect ratio, computed here independently so the
     * preview crop and the encoder crop can be reconciled.
     */
    fun cropRect(upright: Size, aspectW: Int, aspectH: Int, anchor: Anchor = Anchor.CENTER): RectD {
        require(aspectW > 0 && aspectH > 0)
        val target = aspectW.toDouble() / aspectH
        var w = upright.width.toDouble()
        var h = upright.height.toDouble()
        if (w / h > target) {
            w = h * target
        } else {
            h = w / target
        }
        val left = when (anchor) {
            Anchor.LEFT -> 0.0
            Anchor.RIGHT -> upright.width - w
            else -> (upright.width - w) / 2.0
        }
        val top = when (anchor) {
            Anchor.TOP -> 0.0
            Anchor.BOTTOM -> upright.height - h
            else -> (upright.height - h) / 2.0
        }
        return RectD(left, top, left + w, top + h)
    }

    /** Matrix mapping sensor-buffer points into the upright buffer at [relRotationDeg]. */
    fun sensorToUprightMatrix(sensor: Size, relRotationDeg: Int): Mat3 =
        Mat3.clockwise(relRotationDeg, sensor.width.toDouble(), sensor.height.toDouble())

    /** Matrix mapping upright-buffer points back into the sensor buffer. */
    fun uprightToSensorMatrix(sensor: Size, relRotationDeg: Int): Mat3 {
        val upright = uprightSize(sensor, relRotationDeg)
        return Mat3.clockwise(
            360 - Math.floorMod(relRotationDeg, 360),
            upright.width.toDouble(),
            upright.height.toDouble(),
        )
    }

    fun sensorToUprightRect(rect: RectD, sensor: Size, relRotationDeg: Int): RectD =
        sensorToUprightMatrix(sensor, relRotationDeg).map(rect)

    fun uprightToSensorRect(rect: RectD, sensor: Size, relRotationDeg: Int): RectD =
        uprightToSensorMatrix(sensor, relRotationDeg).map(rect)

    /**
     * Direction of a world-fixed arrow (e.g. a printed target pointing to the sky)
     * inside the raw sensor buffer, while the device sits at [displayRotationDeg].
     */
    fun arrowInSensorBuffer(worldArrowDeg: Int, displayRotationDeg: Int, sensorOrientationDeg: Int): Int =
        Math.floorMod(worldArrowDeg + displayRotationDeg - sensorOrientationDeg, 360)

    /**
     * Direction of the arrow on the preview screen. Preview targetRotation tracks the
     * display, so the arrow always appears at its world direction.
     */
    fun arrowOnPreview(worldArrowDeg: Int): Int = Math.floorMod(worldArrowDeg, 360)

    /**
     * Direction of the arrow when the encoded file is played back. The file's rotation
     * metadata is locked to [targetRotationDeg] at record start, so frames captured at
     * [captureRotationDeg] appear rotated by the difference. Independent of the sensor
     * orientation, which is the key invariant of the prototype.
     */
    fun arrowInPlayback(worldArrowDeg: Int, captureRotationDeg: Int, targetRotationDeg: Int): Int =
        Math.floorMod(worldArrowDeg + captureRotationDeg - targetRotationDeg, 360)

    /** Rotation value stored in the encoded file's metadata (e.g. mvhd/tkhd rotation). */
    fun encodedMetadataRotation(sensorOrientationDeg: Int, targetRotationDeg: Int, frontCamera: Boolean = false): Int =
        relativeRotation(sensorOrientationDeg, targetRotationDeg, frontCamera)
}
