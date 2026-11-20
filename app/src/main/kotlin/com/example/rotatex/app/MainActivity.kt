package com.example.rotatex.app

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.rotatex.app.RecordingEngine.Companion.fromSurfaceRotation

/**
 * Prototype screen: CameraX viewfinder + record button + display-area switch +
 * target-orientation picker + reconciliation HUD.
 *
 * The activity handles orientation changes itself (see manifest configChanges), so
 * rotating the device mid-recording never recreates the activity, never rebinds
 * VideoCapture, and never interrupts the audio track.
 */
class MainActivity : ComponentActivity() {

    private lateinit var engine: RecordingEngine
    private lateinit var previewView: PreviewView
    private lateinit var hud: CropHudView
    private lateinit var recordButton: Button

    private val permissions = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine = RecordingEngine(this, this)
        engine.onHud = { state ->
            runOnUiThread {
                hud.update(state)
                recordButton.text = if (state.recording) "停止" else "录制"
            }
        }
        engine.onError = { msg -> runOnUiThread { toast(msg) } }
        setContentView(buildContentView())
        if (hasPermissions()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, permissions, 1)
        }
    }

    private fun buildContentView(): View {
        previewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
        hud = CropHudView(this)
        recordButton = Button(this).apply { text = "录制" }
        val areaButton = Button(this).apply { text = "切换展示区" }
        val targetSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("成片方向 0°", "成片方向 90°", "成片方向 180°", "成片方向 270°"),
            )
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(recordButton)
            addView(areaButton)
            addView(targetSpinner)
        }
        val root = FrameLayout(this)
        root.addView(previewView)
        root.addView(hud)
        root.addView(controls, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { topMargin = 8 })

        recordButton.setOnClickListener { engine.toggleRecording() }
        areaButton.setOnClickListener { engine.cycleDisplayArea() }
        targetSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (engine.isRecording) {
                    toast("录制中锁定成片方向")
                    return
                }
                engine.targetRotationDeg = position * 90
                startCamera()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        return root
    }

    private fun startCamera() {
        val rotation = display?.rotation?.fromSurfaceRotation() ?: 0
        engine.startCamera(previewView.surfaceProvider, rotation)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // Rotation without recreation: rebind only the Preview for the new display
        // rotation; VideoCapture and its audio track keep running.
        val rotation = display?.rotation?.fromSurfaceRotation() ?: return
        engine.onDisplayRotation(rotation)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (hasPermissions()) {
            startCamera()
        } else {
            toast("需要相机和麦克风权限")
        }
    }

    override fun onDestroy() {
        engine.release()
        super.onDestroy()
    }

    private fun hasPermissions(): Boolean = permissions.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
