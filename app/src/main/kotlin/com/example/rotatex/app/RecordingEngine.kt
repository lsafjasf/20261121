package com.example.rotatex.app

import android.content.Context
import android.util.Rational
import android.util.Size as AndroidSize
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.rotatex.Anchor
import com.example.rotatex.CropReconciler
import com.example.rotatex.RecordingMetadata
import com.example.rotatex.RectD
import com.example.rotatex.Size
import com.example.rotatex.TargetTransform
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * CameraX recording engine for the rotation prototype.
 *
 * Design:
 *  - [VideoCapture] is bound once with targetRotation = the selected target
 *    orientation and a ViewPort for the encoder crop. It is NEVER unbound while
 *    recording, so the encoded stream and the audio track continue uninterrupted
 *    when the device rotates.
 *  - [Preview] lives in its own UseCaseGroup. On every display rotation (or
 *    display-area switch) it is unbound and re-bound with a fresh ViewPort whose
 *    crop is derived from the encoder's sensor-space window via [CropReconciler],
 *    so the preview keeps showing exactly what is being encoded.
 *  - Every change is logged to a [RecordingMetadata] sidecar next to the mp4.
 */
class RecordingEngine(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
) {
    /** Display-area presets for the preview. MATCH_ENCODER tracks the encoded window. */
    enum class DisplayArea(val label: String, val aspect: Rational?) {
        MATCH_ENCODER("跟随成片", null),
        FULL("全幅", null),
        WIDE_16_9("16:9", Rational(16, 9)),
        STANDARD_4_3("4:3", Rational(4, 3)),
        SQUARE_1_1("1:1", Rational(1, 1)),
    }

    data class HudState(
        val displayRotationDeg: Int,
        val targetRotationDeg: Int,
        val recording: Boolean,
        val audioActive: Boolean,
        val displayArea: String,
        val previewCropSensor: RectD?,
        val encoderCropSensor: RectD?,
        val report: CropReconciler.Report?,
    )

    var sensorOrientationDeg: Int = 90
        private set
    var sensorSize: Size = Size(1280, 720)
        private set
    var targetRotationDeg: Int = 0
    var displayArea: DisplayArea = DisplayArea.MATCH_ENCODER
        private set

    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var metadata: RecordingMetadata? = null
    private var metadataFile: File? = null
    private var recordStartMs: Long = 0L
    private var displayRotationDeg: Int = 0
    private var surfaceProvider: Preview.SurfaceProvider? = null
    private var encoderSensorCrop: RectD? = null
    private var encoderUprightCrop: RectD? = null

    var onHud: ((HudState) -> Unit)? = null
    var onError: ((String) -> Unit)? = null

    val isRecording: Boolean get() = recording != null

    fun startCamera(surfaceProvider: Preview.SurfaceProvider, displayRotationDeg: Int) {
        this.surfaceProvider = surfaceProvider
        this.displayRotationDeg = displayRotationDeg
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider
            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                val info = provider.availableCameraInfos.firstOrNull()
                if (info != null) {
                    sensorOrientationDeg = info.getSensorRotationDegrees(displayRotationDeg.toSurfaceRotation())
                }
            } catch (_: Exception) {
                // Keep the default 90 deg; emulator cameras usually report 90.
            }
            bindVideoCapture(provider, selector)
            rebindPreview(provider, selector)
            publishHud()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindVideoCapture(provider: ProcessCameraProvider, selector: CameraSelector) {
        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.HD))
            .build()
        val capture = VideoCapture.withOutput(recorder)
        capture.targetRotation = targetRotationDeg.toSurfaceRotation()
        val (upright, sensorCrop) = CropReconciler.encoderWindow(
            sensorSize, sensorOrientationDeg, targetRotationDeg, 16, 9, Anchor.CENTER,
        )
        encoderUprightCrop = upright
        encoderSensorCrop = sensorCrop
        val group = UseCaseGroup.Builder()
            .addUseCase(capture)
            .setViewPort(ViewPort.Builder(Rational(16, 9), targetRotationDeg.toSurfaceRotation()).build())
            .build()
        provider.unbind(videoCapture)
        videoCapture = capture
        provider.bindToLifecycle(lifecycleOwner, selector, group)
    }

    /** Rebinds ONLY the Preview use case; VideoCapture keeps running (audio included). */
    private fun rebindPreview(provider: ProcessCameraProvider, selector: CameraSelector) {
        val newPreview = Preview.Builder()
            .setTargetRotation(displayRotationDeg.toSurfaceRotation())
            .build()
            .also { it.setSurfaceProvider(surfaceProvider) }
        val aspect = previewAspect()
        val groupBuilder = UseCaseGroup.Builder().addUseCase(newPreview)
        if (aspect != null) {
            groupBuilder.setViewPort(ViewPort.Builder(aspect, displayRotationDeg.toSurfaceRotation()).build())
        }
        provider.unbind(preview)
        preview = newPreview
        provider.bindToLifecycle(lifecycleOwner, selector, groupBuilder.build())
    }

    /** Aspect for the preview ViewPort; null means no ViewPort (full frame). */
    private fun previewAspect(): Rational? = when (displayArea) {
        DisplayArea.MATCH_ENCODER -> encoderUprightCrop?.let {
            Rational(it.width.toInt(), it.height.toInt())
        }
        DisplayArea.FULL -> null
        else -> displayArea.aspect
    }

    fun onDisplayRotation(deg: Int) {
        if (deg == displayRotationDeg) return
        displayRotationDeg = deg
        val provider = cameraProvider ?: return
        rebindPreview(provider, CameraSelector.DEFAULT_BACK_CAMERA)
        logEvent(RecordingMetadata.Event.Type.ROTATION)
        publishHud()
    }

    fun cycleDisplayArea() {
        val values = DisplayArea.entries
        displayArea = values[(displayArea.ordinal + 1) % values.size]
        val provider = cameraProvider ?: return
        rebindPreview(provider, CameraSelector.DEFAULT_BACK_CAMERA)
        logEvent(RecordingMetadata.Event.Type.DISPLAY_AREA_CHANGE)
        publishHud()
    }

    fun toggleRecording() {
        if (recording != null) stopRecording() else startRecording()
    }

    private fun startRecording() {
        val capture = videoCapture ?: return
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = context.getExternalFilesDir(android.os.Environment.DIRECTORY_MOVIES) ?: context.filesDir
        val videoFile = File(dir, "rotatex_$stamp.mp4")
        metadataFile = File(dir, "rotatex_$stamp.json")
        metadata = RecordingMetadata(
            sensorWidth = sensorSize.width,
            sensorHeight = sensorSize.height,
            sensorOrientationDeg = sensorOrientationDeg,
            targetRotationDeg = targetRotationDeg,
            encodedMetadataRotationDeg = TargetTransform.encodedMetadataRotation(sensorOrientationDeg, targetRotationDeg),
        )
        recordStartMs = System.currentTimeMillis()
        val pending = capture.output
            .prepareRecording(context, FileOutputOptions.Builder(videoFile).build())
            .withAudioEnabled()
        recording = pending.start(ContextCompat.getMainExecutor(context)) { event ->
            if (event is VideoRecordEvent.Finalize) {
                if (event.hasError()) {
                    metadata?.audioInterrupted = true
                    onError?.invoke("recording finalize error ${event.error}")
                }
                metadataFile?.writeText(metadata?.toJson().orEmpty())
                publishHud()
            }
        }
        logEvent(RecordingMetadata.Event.Type.RECORD_START)
        publishHud()
    }

    private fun stopRecording() {
        logEvent(RecordingMetadata.Event.Type.RECORD_STOP)
        recording?.stop()
        recording = null
        publishHud()
    }

    private fun currentPreviewSensorCrop(): RectD? {
        val encoder = encoderSensorCrop ?: return null
        return when (displayArea) {
            DisplayArea.MATCH_ENCODER -> encoder
            DisplayArea.FULL -> RectD(0.0, 0.0, sensorSize.width.toDouble(), sensorSize.height.toDouble())
            else -> {
                val rel = TargetTransform.relativeRotation(sensorOrientationDeg, displayRotationDeg)
                val upright = TargetTransform.uprightSize(sensorSize, rel)
                val aspect = displayArea.aspect ?: return null
                val crop = TargetTransform.cropRect(upright, aspect.numerator, aspect.denominator)
                TargetTransform.uprightToSensorRect(crop, sensorSize, rel)
            }
        }
    }

    private fun logEvent(type: RecordingMetadata.Event.Type) {
        val meta = metadata ?: return
        val encoder = encoderSensorCrop ?: return
        val previewCrop = currentPreviewSensorCrop() ?: return
        meta.events += RecordingMetadata.Event(
            tMs = System.currentTimeMillis() - recordStartMs,
            type = type,
            displayRotationDeg = displayRotationDeg,
            previewCropSensor = previewCrop.toList(),
            encoderCropSensor = encoder.toList(),
            cropsMatch = previewCrop.maxDeviation(encoder) <= 1.0,
        )
    }

    private fun publishHud() {
        val encoder = encoderSensorCrop
        val previewCrop = currentPreviewSensorCrop()
        val report = if (encoder != null && previewCrop != null) {
            CropReconciler.Report(previewCrop, encoder, previewCrop.maxDeviation(encoder), 1.0)
        } else {
            null
        }
        onHud?.invoke(
            HudState(
                displayRotationDeg = displayRotationDeg,
                targetRotationDeg = targetRotationDeg,
                recording = isRecording,
                audioActive = isRecording,
                displayArea = displayArea.label,
                previewCropSensor = previewCrop,
                encoderCropSensor = encoder,
                report = report,
            ),
        )
    }

    fun release() {
        recording?.stop()
        recording = null
        cameraProvider?.unbindAll()
        cameraProvider = null
    }

    companion object {
        fun Int.toSurfaceRotation(): Int = when (Math.floorMod(this, 360)) {
            0 -> android.view.Surface.ROTATION_0
            90 -> android.view.Surface.ROTATION_90
            180 -> android.view.Surface.ROTATION_180
            else -> android.view.Surface.ROTATION_270
        }

        fun Int.fromSurfaceRotation(): Int = when (this) {
            android.view.Surface.ROTATION_90 -> 90
            android.view.Surface.ROTATION_180 -> 180
            android.view.Surface.ROTATION_270 -> 270
            else -> 0
        }
    }
}
