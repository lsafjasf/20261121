/*
 * fade_router — JACK 处理客户端
 *
 * 功能:
 *   - 两路独立输出: 监听耳机 headphones 与场记录音 recorder(分走);
 *   - 自动发现名字匹配 "<prefix>*:out" 的设备端口(默认 prefix=vdev);
 *   - 新设备出现时以短淡化(默认 10ms)把当前源迁移到新设备, 无爆音;
 *   - 设备消失时 JACK 自动断开, 输出自然静音; 重新出现时再次淡化接入。
 *
 * 实时安全: process 回调内不取锁、不访问磁盘、不做动态内存分配。
 * 跨线程通信全部使用原子变量与预分配的 SPSC 无锁环。
 */
#include <jack/jack.h>

#include <stdatomic.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_NAME 64
#define REG_RING 64

typedef struct {
    char name[MAX_NAME];
} reg_event_t;

static jack_client_t *client;
static jack_port_t *in_ports[2];
static jack_port_t *out_hp;
static jack_port_t *out_rec;

/* ---- RT 线程私有状态(仅 process 回调访问) ---- */
static int rt_active = 0;              /* 当前输入槽位: 0/1 */
static int rt_fading = 0;
static int rt_target = 0;
static jack_nframes_t rt_fade_pos = 0;
static jack_nframes_t rt_fade_len = 480; /* main 中按采样率重算 */

/* ---- 无锁跨线程通信 ---- */
static _Atomic int pending_slot = 0;   /* 主->RT: 请求迁移到的槽位+1, 0=无 */
static _Atomic int done_slot = 0;      /* RT->主: 淡化完成, 待断开的旧槽位+1 */
static _Atomic int pub_active = 1;     /* RT->主: 当前槽位+1(显示用) */
static _Atomic int pub_fading = 0;     /* RT->主: 是否淡化中(显示用) */
static _Atomic unsigned long xruns = 0;
static _Atomic unsigned long fades = 0;

/* 端口注册事件环: 注册回调(JACK 通知线程)写, 主线程读 */
static reg_event_t reg_ring[REG_RING];
static _Atomic unsigned reg_head;      /* 写指针 */
static _Atomic unsigned reg_tail;      /* 读指针 */

static char prefix[32] = "vdev";
static volatile sig_atomic_t quit_flag;

static void on_signal(int sig)
{
    (void)sig;
    quit_flag = 1;
}

static void on_port_reg(jack_port_id_t port, int is_reg, void *arg)
{
    (void)arg;
    if (!is_reg)
        return;
    jack_port_t *p = jack_port_by_id(client, port);
    if (!p)
        return;
    const char *name = jack_port_name(p);
    if (strncmp(name, prefix, strlen(prefix)) != 0 || !strstr(name, ":out"))
        return;
    unsigned head = atomic_load_explicit(&reg_head, memory_order_relaxed);
    unsigned next = (head + 1) % REG_RING;
    if (next == atomic_load_explicit(&reg_tail, memory_order_acquire))
        return; /* 环满丢弃: 下一次注册事件仍会到来 */
    snprintf(reg_ring[head].name, MAX_NAME, "%s", name);
    atomic_store_explicit(&reg_head, next, memory_order_release);
}

static int on_process(jack_nframes_t n, void *arg)
{
    (void)arg;
    float *src[2] = {
        jack_port_get_buffer(in_ports[0], n),
        jack_port_get_buffer(in_ports[1], n),
    };
    float *hp = jack_port_get_buffer(out_hp, n);
    float *rec = jack_port_get_buffer(out_rec, n);

    int pend = atomic_exchange_explicit(&pending_slot, 0, memory_order_acq_rel);
    if (pend != 0) {
        if (!rt_fading) {
            rt_fading = 1;
            rt_target = pend - 1;
            rt_fade_pos = 0;
            atomic_fetch_add_explicit(&fades, 1, memory_order_relaxed);
        } else {
            atomic_store_explicit(&pending_slot, pend, memory_order_release);
        }
    }

    for (jack_nframes_t i = 0; i < n; i++) {
        float s;
        if (rt_fading) {
            float g = (float)rt_fade_pos / (float)rt_fade_len;
            s = src[rt_active][i] * (1.0f - g) + src[rt_target][i] * g;
            if (++rt_fade_pos >= rt_fade_len) {
                int old = rt_active;
                rt_active = rt_target;
                rt_fading = 0;
                atomic_store_explicit(&done_slot, old + 1, memory_order_release);
                atomic_store_explicit(&pub_active, rt_active + 1, memory_order_release);
            }
        } else {
            s = src[rt_active][i];
        }
        hp[i] = s;  /* 监听耳机 */
        rec[i] = s; /* 场记录音 */
    }
    atomic_store_explicit(&pub_fading, rt_fading, memory_order_relaxed);
    return 0;
}

static int on_xrun(void *arg)
{
    (void)arg;
    atomic_fetch_add_explicit(&xruns, 1, memory_order_relaxed);
    return 0;
}

int main(int argc, char **argv)
{
    const char *client_name = "fade_router";
    int fade_ms = 10;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prefix") && i + 1 < argc) {
            snprintf(prefix, sizeof(prefix), "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--fade-ms") && i + 1 < argc) {
            fade_ms = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--name") && i + 1 < argc) {
            client_name = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--prefix vdev] [--fade-ms 10] [--name fade_router]\n", argv[0]);
            return 2;
        }
    }

    client = jack_client_open(client_name, JackNullOption, NULL);
    if (!client) {
        fprintf(stderr, "[fade_router] cannot connect to JACK server\n");
        return 1;
    }
    rt_fade_len = (jack_nframes_t)((unsigned long)jack_get_sample_rate(client) *
                                   (unsigned)(fade_ms > 0 ? fade_ms : 1) / 1000ul);
    if (rt_fade_len == 0)
        rt_fade_len = 1;

    in_ports[0] = jack_port_register(client, "in_a", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    in_ports[1] = jack_port_register(client, "in_b", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    out_hp = jack_port_register(client, "headphones", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    out_rec = jack_port_register(client, "recorder", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (!in_ports[0] || !in_ports[1] || !out_hp || !out_rec) {
        fprintf(stderr, "[fade_router] port register failed\n");
        return 1;
    }

    jack_set_process_callback(client, on_process, NULL);
    jack_set_xrun_callback(client, on_xrun, NULL);
    jack_set_port_registration_callback(client, on_port_reg, NULL);

    if (jack_activate(client) != 0) {
        fprintf(stderr, "[fade_router] cannot activate client\n");
        return 1;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    /* 主线程(非 RT)状态: 设备名簿 + 迁移串行化 */
    static char slot_dev[2][MAX_NAME]; /* 每个输入槽位当前连接的设备 */
    char queued_dev[MAX_NAME] = "";    /* 等待迁移的最新设备 */
    int migrating = 0;
    int tty = isatty(STDOUT_FILENO);
    int tick = 0;

    fprintf(stderr, "[fade_router] ready: in_a/in_b -> headphones+recorder, fade=%u samples (%d ms)\n",
            (unsigned)rt_fade_len, fade_ms);

    while (!quit_flag) {
        /* 1. 收取注册事件, 只保留最新设备名(新设备优先) */
        unsigned tail = atomic_load_explicit(&reg_tail, memory_order_relaxed);
        unsigned head = atomic_load_explicit(&reg_head, memory_order_acquire);
        int got = 0;
        while (tail != head) {
            snprintf(queued_dev, MAX_NAME, "%s", reg_ring[tail].name);
            tail = (tail + 1) % REG_RING;
            got = 1;
        }
        atomic_store_explicit(&reg_tail, tail, memory_order_release);

        /* 2. 发起迁移: 把新设备接到空闲槽位, 通知 RT 线程开始淡化 */
        if (got && !migrating && queued_dev[0] != '\0') {
            int idle = (atomic_load_explicit(&pub_active, memory_order_acquire) - 1) ^ 1;
            if (jack_connect(client, queued_dev, jack_port_name(in_ports[idle])) == 0) {
                snprintf(slot_dev[idle], MAX_NAME, "%s", queued_dev);
                atomic_store_explicit(&pending_slot, idle + 1, memory_order_release);
                migrating = 1;
                fprintf(stderr, "[fade_router] migrate -> %s (fade %d ms)\n", queued_dev, fade_ms);
            } else {
                fprintf(stderr, "[fade_router] connect failed: %s (device gone?)\n", queued_dev);
            }
            queued_dev[0] = '\0';
        }

        /* 3. 淡化完成: 断开旧槽位上的设备 */
        int done = atomic_exchange_explicit(&done_slot, 0, memory_order_acq_rel);
        if (done != 0) {
            int old = done - 1;
            jack_port_disconnect(client, in_ports[old]);
            slot_dev[old][0] = '\0';
            migrating = 0;
        }

        /* 4. 前端: 显示 xrun 与当前路由 */
        int act = atomic_load_explicit(&pub_active, memory_order_acquire) - 1;
        const char *route = slot_dev[act][0] ? slot_dev[act] : "(none)";
        const char *state = atomic_load_explicit(&pub_fading, memory_order_relaxed)
                                ? "FADING" : "STABLE";
        if (tty) {
            printf("\r[fade_router] xruns=%lu fades=%lu route=%s -> [headphones, recorder] %-6s ",
                   atomic_load(&xruns), atomic_load(&fades), route, state);
            fflush(stdout);
        } else if (++tick >= 10) { /* 非 tty: 每 ~1s 打一行日志 */
            tick = 0;
            printf("[fade_router] xruns=%lu fades=%lu route=%s -> [headphones, recorder] %s\n",
                   atomic_load(&xruns), atomic_load(&fades), route, state);
            fflush(stdout);
        }
        usleep(100000);
    }

    jack_client_close(client);
    if (tty)
        printf("\n");
    fprintf(stderr, "[fade_router] exit: total xruns=%lu fades=%lu\n",
            atomic_load(&xruns), atomic_load(&fades));
    return 0;
}
