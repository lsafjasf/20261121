package com.example.rotatex.sim

import com.example.rotatex.RecordingMetadata
import com.example.rotatex.Size
import java.io.File
import javax.imageio.ImageIO

/**
 * Emulator-free demo of the recording-rotation prototype. Renders the arrow test
 * target, walks the device through all four rotations "while recording", and writes:
 *
 *   demo_out/target.png            printable arrow target (world-up)
 *   demo_out/sensor_R*.png         raw sensor frames per display rotation
 *   demo_out/preview_R*.png        what the preview shows (always upright)
 *   demo_out/encoded_R*.png        what the encoder stores (locked window)
 *   demo_out/playback_R*.png       what a player shows (file keeps target orientation)
 *   demo_out/metadata.json         sidecar metadata for reconciliation
 */
fun main(args: Array<String>) {
    val outDir = File(args.getOrElse(0) { "demo_out" })
    outDir.mkdirs()

    val sim = Simulator()
    ImageIO.write(ArrowTarget.render(Size(720, 1280), 0), "png", File(outDir, "target.png"))

    val rotations = intArrayOf(0, 90, 180, 270)
    rotations.forEachIndexed { i, r ->
        val type = if (i == 0) {
            RecordingMetadata.Event.Type.RECORD_START
        } else {
            RecordingMetadata.Event.Type.ROTATION
        }
        val frame = sim.frameAt(r, tMs = i * 1_000L, type = type)
        ImageIO.write(frame.preview, "png", File(outDir, "preview_R$r.png"))
        ImageIO.write(frame.encoded, "png", File(outDir, "encoded_R$r.png"))
        ImageIO.write(frame.playback, "png", File(outDir, "playback_R$r.png"))
        val arrow = ArrowTarget.detectArrowDirection(frame.playback)
        println(
            "R=$r preview=${frame.preview.width}x${frame.preview.height} " +
                "playbackArrow=${arrow}deg cropsMatch=${frame.cropsMatch}",
        )
    }
    sim.metadata.events += RecordingMetadata.Event(
        tMs = 4_000,
        type = RecordingMetadata.Event.Type.RECORD_STOP,
        displayRotationDeg = 0,
        previewCropSensor = sim.metadata.events.last().previewCropSensor,
        encoderCropSensor = sim.metadata.events.last().encoderCropSensor,
        cropsMatch = true,
    )
    File(outDir, "metadata.json").writeText(sim.metadata.toJson())

    val problems = sim.metadata.validate()
    if (problems.isEmpty()) {
        println("metadata OK: ${sim.metadata.events.size} events, audio continuous, crops reconciled")
    } else {
        println("metadata problems:")
        problems.forEach { println("  - $it") }
    }
    println("demo output written to ${outDir.absolutePath}")
}
