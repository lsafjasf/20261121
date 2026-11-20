# JACK 淡化迁移客户端(fade_router)

一个可运行的 JACK 处理客户端:监听耳机与场记录音分路输出,新设备接入时以
短淡化完成迁移,音频回调内无锁、无磁盘、无动态分配。附带虚拟设备模拟器
(设备消失 / 松动重连)与录音样本顺序校验。

## 组成

- `src/fade_router.c` — 处理客户端 + 前端(状态行显示 xrun 与当前路由)
- `src/virtual_device.c` — 虚拟设备,输出递增斜坡测试信号,可模拟消失/松动
- `src/recorder.c` — 把 `fade_router:recorder` 录为 float32 raw(磁盘写在非 RT 线程)
- `scripts/verify_recording.py` — 校验录音的有效输出与样本顺序
- `run_demo.sh` — 一键端到端演示(自动起 dummy 后端的 jackd)
- `.jack/` — 本地解压的 jackd2/libjack(无 root 环境用,`JACK_DRIVER_DIR` 指定后端目录)

## 构建与运行

```sh
make                 # 需要 jack 头文件/库, 默认取 .jack/root(本地解压的 jackd2)
./run_demo.sh        # 场景1: 接力迁移+设备消失; 场景2: 松动反复重连
```

前台交互运行客户端(tty 下状态行实时刷新):

```sh
./build/fade_router --fade-ms 10 --prefix vdev
```

## 实时安全设计

- `process` 回调只做:取端口缓冲、按淡化增益混音、写两路输出;
  不取锁、不分配内存、不访问磁盘、不调用 `jack_connect`。
- 端口注册回调(JACK 通知线程)→ 预分配 SPSC 无锁环 → 主线程;
  主线程完成 `jack_connect` 后经原子变量 `pending_slot` 通知 RT 线程开始淡化。
- RT 线程淡化完成后经原子变量 `done_slot` 通知主线程断开旧设备。
- 双输入槽位 `in_a`/`in_b`:新旧设备各接一路,线性短淡化(默认 10 ms)切换。

## 验证方法

虚拟设备输出严格递增斜坡(每样本 +2.5e-6)。校验器确认:

- 有效输出:样本全部有限、幅度 ≤ 1、非全静音;
- 样本顺序:每个非静音片段中段差分恒等于步进(无丢样/重复/乱序);
- 淡化迁移表现为少量短促"过渡带"(≤ 1200 样本),过渡带之外零异常。
