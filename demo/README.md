# 模拟器演示（四方向 × 箭头目标 × 元数据核对）

## 准备

1. 启动 Android 模拟器（API 26+，带后置摄像头；虚拟场景或 webcam 均可）。
2. 在电脑屏幕或另一台设备上全屏显示箭头测试目标：先运行
   `./gradlew :simulator:run` 生成 `demo_out/target.png`（箭头指向世界"上"方，
   四边有不同颜色标记），把它对准摄像头。
3. 安装应用：`./gradlew :app:installDebug`，打开 "RotateX Recorder"。
4. 选择"成片方向"（例如 0°），点击 **录制**。

## 演示步骤

```bash
./demo/rotate_emulator.sh
```

脚本依次把模拟器强制旋转到 0°/90°/180°/270°，每个方向截图，最后把录制的
mp4 与 sidecar JSON 元数据拉到 `demo/emulator_out/`。

也可以手动旋转：模拟器侧边栏旋转按钮，或
`adb shell settings put system user_rotation <0..3>`。

录制中可随时点 **切换展示区**（跟随成片 / 全幅 / 16:9 / 4:3 / 1:1），HUD 实时
显示预览裁切与编码裁切（sensor 坐标）以及 MATCH/MISMATCH 核对结果。

## 验收要点

| 项目 | 预期 |
| --- | --- |
| 预览 | 四个方向下箭头始终在屏幕上指"上"，预览跟随旋转 |
| 成片 | 整段视频保持录制时选定的目标方向；设备转动期间画面随之旋转（文件旋转元数据恒定） |
| 音频 | `ffprobe` 只有 1 条连续音频流，旋转时刻无中断 |
| 元数据 | sidecar JSON 中 `encodedMetadataRotationDeg` 恒定，旋转事件 `cropsMatch=true`（展示区=跟随成片时） |
| 核对 | HUD 在"跟随成片"时显示 MATCH；切到其它展示区时如实显示 MISMATCH |

## 无设备验证

没有模拟器时，`./gradlew :simulator:run` 在 JVM 里模拟同一套管线，输出
`demo_out/preview_R*.png`、`encoded_R*.png`、`playback_R*.png` 与
`metadata.json`；`./gradlew :simulator:test` 对四个方向做像素级箭头方向断言。
