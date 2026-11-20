package com.example.rotatex

import kotlin.math.abs

/** Integer pixel size of a buffer. */
data class Size(val width: Int, val height: Int) {
    val aspect: Double get() = width.toDouble() / height
    fun rotated(): Size = Size(height, width)
    override fun toString(): String = "${width}x$height"
}

/** Continuous rect in buffer coordinates (origin top-left, x right, y down). */
data class RectD(val left: Double, val top: Double, val right: Double, val bottom: Double) {
    val width: Double get() = right - left
    val height: Double get() = bottom - top
    val centerX: Double get() = (left + right) / 2.0
    val centerY: Double get() = (top + bottom) / 2.0
    val aspect: Double get() = width / height

    fun toList(): List<Double> = listOf(left, top, right, bottom)

    fun maxDeviation(other: RectD): Double = maxOf(
        abs(left - other.left), abs(top - other.top),
        abs(right - other.right), abs(bottom - other.bottom),
    )

    override fun toString(): String = "[%.1f, %.1f, %.1f, %.1f]".format(left, top, right, bottom)
}

/** Row-major 3x3 homogeneous 2D transform. Pure Kotlin; mirrors android.graphics.Matrix semantics. */
class Mat3(val m: DoubleArray) {
    init {
        require(m.size == 9) { "Mat3 needs 9 elements" }
    }

    operator fun times(o: Mat3): Mat3 {
        val r = DoubleArray(9)
        for (row in 0..2) {
            for (col in 0..2) {
                r[row * 3 + col] =
                    m[row * 3] * o.m[col] +
                    m[row * 3 + 1] * o.m[3 + col] +
                    m[row * 3 + 2] * o.m[6 + col]
            }
        }
        return Mat3(r)
    }

    fun map(x: Double, y: Double): Pair<Double, Double> {
        val w = m[6] * x + m[7] * y + m[8]
        return Pair(
            (m[0] * x + m[1] * y + m[2]) / w,
            (m[3] * x + m[4] * y + m[5]) / w,
        )
    }

    /** Maps the four corners of [r] and returns their bounding box. */
    fun map(r: RectD): RectD {
        val xs = doubleArrayOf(r.left, r.right, r.right, r.left)
        val ys = doubleArrayOf(r.top, r.top, r.bottom, r.bottom)
        var minX = Double.POSITIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY
        for (i in 0..3) {
            val (px, py) = map(xs[i], ys[i])
            if (px < minX) minX = px
            if (py < minY) minY = py
            if (px > maxX) maxX = px
            if (py > maxY) maxY = py
        }
        return RectD(minX, minY, maxX, maxY)
    }

    companion object {
        fun identity(): Mat3 = Mat3(
            doubleArrayOf(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ),
        )

        /**
         * Transform that maps points of a [srcWidth]x[srcHeight] buffer into the buffer
         * produced by rotating it clockwise by [deg] (0/90/180/270).
         */
        fun clockwise(deg: Int, srcWidth: Double, srcHeight: Double): Mat3 = when (Math.floorMod(deg, 360)) {
            0 -> identity()
            90 -> Mat3(
                doubleArrayOf(
                    0.0, -1.0, srcHeight,
                    1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0,
                ),
            )
            180 -> Mat3(
                doubleArrayOf(
                    -1.0, 0.0, srcWidth,
                    0.0, -1.0, srcHeight,
                    0.0, 0.0, 1.0,
                ),
            )
            270 -> Mat3(
                doubleArrayOf(
                    0.0, 1.0, 0.0,
                    -1.0, 0.0, srcWidth,
                    0.0, 0.0, 1.0,
                ),
            )
            else -> throw IllegalArgumentException("deg must be a multiple of 90: $deg")
        }
    }
}
