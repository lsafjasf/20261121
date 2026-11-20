/*
 * recorder — 场记录音采集端
 *
 * 把指定 JACK 端口(默认 fade_router:recorder)录为 float32 raw 文件。
 * RT 回调只写预分配的无锁 SPSC 环; 磁盘写由主线程完成。
 *
 * usage: recorder <outfile.raw> <seconds> [source_port]
 */
#include <jack/jack.h>

#include <stdatomic.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RING_BITS 22
#define RING_SIZE (1ul << RING_BITS) /* 4M floats = 16MB, 约 87s @48kHz */
#define RING_MASK (RING_SIZE - 1)

static float ring[RING_SIZE];          /* 静态预分配, 无动态内存 */
static _Atomic unsigned long r_head;   /* RT 线程写 */
static _Atomic unsigned long r_tail;   /* 主线程写 */
static _Atomic unsigned long overruns;
static volatile sig_atomic_t quit_flag;

static void on_signal(int sig)
{
    (void)sig;
    quit_flag = 1;
}

static int on_process(jack_nframes_t n, void *arg)
{
    float *in = jack_port_get_buffer((jack_port_t *)arg, n);
    unsigned long head = atomic_load_explicit(&r_head, memory_order_relaxed);
    unsigned long tail = atomic_load_explicit(&r_tail, memory_order_acquire);
    unsigned long free_n = RING_SIZE - 1 - (head - tail);
    if (n > free_n) {
        atomic_fetch_add_explicit(&overruns, 1, memory_order_relaxed);
        n = free_n;
    }
    for (jack_nframes_t i = 0; i < n; i++)
        ring[(head + i) & RING_MASK] = in[i];
    atomic_store_explicit(&r_head, head + n, memory_order_release);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <outfile.raw> <seconds> [source_port]\n", argv[0]);
        return 2;
    }
    const char *outfile = argv[1];
    double seconds = atof(argv[2]);
    const char *source = argc > 3 ? argv[3] : "fade_router:recorder";

    jack_client_t *client = jack_client_open("rec", JackNullOption, NULL);
    if (!client) {
        fprintf(stderr, "[rec] cannot connect to JACK server\n");
        return 1;
    }
    jack_port_t *in = jack_port_register(client, "in", JACK_DEFAULT_AUDIO_TYPE,
                                         JackPortIsInput, 0);
    if (!in) {
        fprintf(stderr, "[rec] port register failed\n");
        return 1;
    }
    jack_set_process_callback(client, on_process, in);
    if (jack_activate(client) != 0) {
        fprintf(stderr, "[rec] cannot activate client\n");
        return 1;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    /* 等待并连接源端口 */
    while (!quit_flag) {
        if (jack_connect(client, source, jack_port_name(in)) == 0)
            break;
        usleep(100000);
    }
    fprintf(stderr, "[rec] connected: %s -> rec:in, recording %.1fs to %s\n",
            source, seconds, outfile);

    FILE *fp = fopen(outfile, "wb");
    if (!fp) {
        fprintf(stderr, "[rec] cannot open %s\n", outfile);
        return 1;
    }

    unsigned long written = 0;
    long slices = (long)(seconds * 50.0);
    for (long i = 0; i < slices && !quit_flag; i++) {
        usleep(20000);
        unsigned long head = atomic_load_explicit(&r_head, memory_order_acquire);
        unsigned long tail = atomic_load_explicit(&r_tail, memory_order_relaxed);
        /* 分两段处理环绕 */
        while (tail != head) {
            unsigned long idx = tail & RING_MASK;
            unsigned long chunk = head - tail;
            unsigned long to_edge = RING_SIZE - idx;
            if (chunk > to_edge)
                chunk = to_edge;
            written += fwrite(&ring[idx], sizeof(float), chunk, fp);
            tail += chunk;
        }
        atomic_store_explicit(&r_tail, tail, memory_order_release);
    }
    /* 收尾排空 */
    {
        unsigned long head = atomic_load_explicit(&r_head, memory_order_acquire);
        unsigned long tail = atomic_load_explicit(&r_tail, memory_order_relaxed);
        while (tail != head) {
            unsigned long idx = tail & RING_MASK;
            unsigned long chunk = head - tail;
            unsigned long to_edge = RING_SIZE - idx;
            if (chunk > to_edge)
                chunk = to_edge;
            written += fwrite(&ring[idx], sizeof(float), chunk, fp);
            tail += chunk;
        }
        atomic_store_explicit(&r_tail, tail, memory_order_release);
    }
    fclose(fp);
    jack_client_close(client);
    fprintf(stderr, "[rec] wrote %lu frames, ring overruns=%lu\n",
            written, atomic_load(&overruns));
    return 0;
}
