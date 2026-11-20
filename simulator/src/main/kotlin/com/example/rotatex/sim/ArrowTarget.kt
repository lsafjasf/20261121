package com.example.rotatex.sim

import com.example.rotatex.Size
import java.awt.Color
import java.awt.Polygon
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import kotlin.math.cos
import kotlin.math.sin

/**
 * Renders the arrow test target into a simulated sensor frame.
 *
 * The target carries two kinds of orientation evidence:
 *  - a red arrow whose tip sits near the edge the arrow points to, and
 *  - four distinct edge-marker colors (top/right/bottom/left) so tests can read the
 *    absolute rotation of any produced image, not just the arrow direction.
 */
object ArrowTarget {

    val ARROW: Color = Color(220, 30, 30)
    val EDGE_TOP: Color = Color(255, 0, 255) // magenta
    val EDGE_RIGHT: Color = Color(0, 200, 0) // green
    val EDGE_BOTTOM: Color = Color(0, 255, 255) // cyan
    val EDGE_LEFT: Color = Color(255, 220, 0) // yellow
    val BACKGROUND: Color = Color(40, 40, 48)
    val RING: Color = Color(230, 230, 230)

    const val MARKER_FRACTION = 0.04

    /**
     * Draws the target into a [size] buffer with the arrow pointing at
     * [arrowDegInBuffer] (degrees clockwise from the buffer's top edge).
     */
    fun render(size: Size, arrowDegInBuffer: Int): BufferedImage {
        val img = BufferedImage(size.width, size.height, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.color = BACKGROUND
        g.fillRect(0, 0, size.width, size.height)

        val band = (minOf(size.width, size.height) * MARKER_FRACTION).toInt().coerceAtLeast(2)
        g.color = EDGE_TOP
        g.fillRect(0, 0, size.width, band)
        g.color = EDGE_BOTTOM
        g.fillRect(0, size.height - band, size.width, band)
        g.color = EDGE_LEFT
        g.fillRect(0, 0, band, size.height)
        g.color = EDGE_RIGHT
        g.fillRect(size.width - band, 0, band, size.height)

        val cx = size.width / 2.0
        val cy = size.height / 2.0
        val radius = minOf(size.width, size.height) * 0.30
        g.color = RING
        g.drawOval((cx - radius).toInt(), (cy - radius).toInt(), (radius * 2).toInt(), (radius * 2).toInt())

        val rad = Math.toRadians(arrowDegInBuffer.toDouble())
        val dx = sin(rad)
        val dy = -cos(rad)
        val reach = minOf(size.width, size.height) * 0.46
        val tipX = cx + dx * reach
        val tipY = cy + dy * reach
        val baseX = cx + dx * reach * 0.62
        val baseY = cy + dy * reach * 0.62
        val headHalf = minOf(size.width, size.height) * 0.075
        val head = Polygon()
        head.addPoint(tipX.toInt(), tipY.toInt())
        head.addPoint((baseX - dy * headHalf).toInt(), (baseY + dx * headHalf).toInt())
        head.addPoint((baseX + dy * headHalf).toInt(), (baseY - dx * headHalf).toInt())
        g.color = ARROW
        g.fillPolygon(head)
        g.stroke = java.awt.BasicStroke((minOf(size.width, size.height) * 0.03).toFloat())
        g.drawLine(cx.toInt(), cy.toInt(), baseX.toInt(), baseY.toInt())
        g.dispose()
        return img
    }

    /** Direction (deg CW from top) the arrow points to in [img], found by red mass per edge band. */
    fun detectArrowDirection(img: BufferedImage): Int {
        val w = img.width
        val h = img.height
        val band = (minOf(w, h) * 0.25).toInt()
        var top = 0
        var bottom = 0
        var left = 0
        var right = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val c = Color(img.getRGB(x, y))
                val isArrow = c.red > 150 && c.green < 90 && c.blue < 90
                if (!isArrow) continue
                when {
                    y < band -> top++
                    y >= h - band -> bottom++
                    x < band -> left++
                    x >= w - band -> right++
                }
            }
        }
        val counts = intArrayOf(top, right, bottom, left)
        val best = counts.indices.maxByOrNull { counts[it] }!!
        return best * 90
    }

    /** Reads the marker color at the center of each edge of [img]; returns edge->color name. */
    fun detectEdgeMarkers(img: BufferedImage): Map<String, Color> {
        val w = img.width
        val h = img.height
        val inset = (minOf(w, h) * MARKER_FRACTION / 2).toInt().coerceAtLeast(1)
        return mapOf(
            "top" to Color(img.getRGB(w / 2, inset)),
            "right" to Color(img.getRGB(w - 1 - inset, h / 2)),
            "bottom" to Color(img.getRGB(w / 2, h - 1 - inset)),
            "left" to Color(img.getRGB(inset, h / 2)),
        )
    }
}
