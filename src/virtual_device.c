/*
 * virtual_device — 虚拟现场设备
 *
 * 输出严格递增的斜坡测试信号(每样本 +RAMP_STEP), 用于校验录音样本顺序。
 * 通过注销/重新注册输出端口模拟:
 *   stable    <name> <seconds>                 稳定在线
 *   disappear <name> <on_s> <off_s> <cycles>   周期性消失(设备断电/拔走)
 *   flaky     <name> <seconds> <min_ms> <max_ms> [seed]   松动反复重连
 *
 * 端口被注销期间计数器暂停(模拟设备断电, 不产出样本)。
 */
#include <jack/jack.h>

#include <stdatomic.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RAMP_MIN (-0.6f)
#define RAMP_MAX 0.6f
#define RAMP_STEP (1.0f / 400000.0f) /* 2.5e-6: 10 秒扫满 -0.6..0.6 @48kHz */
#define COUNTER_MAX 480000ull        /* (RAMP_MAX-RAMP_MIN)/RAMP_STEP */

static jack_client_t *client;
static _Atomic(jack_port_t *) out_port; /* main 注册/注销, RT 线程读取 */
static volatile sig_atomic_t quit_flag;
static unsigned long long counter;      /* 仅 RT 线程访问 */

static void on_signal(int sig)
{
    (void)sig;
    quit_flag = 1;
}

static int on_process(jack_nframes_t n, void *arg)
{
    (void)arg;
    jack_port_t *p = atomic_load_explicit(&out_port, memory_order_acquire);
    if (!p)
        return 0; /* 设备"断电": 无端口, 不产出样本 */
    float *out = jack_port_get_buffer(p, n);
    if (!out)
        return 0;
    for (jack_nframes_t i = 0; i < n; i++) {
        float v = RAMP_MIN + (float)counter * RAMP_STEP;
        out[i] = v > RAMP_MAX ? RAMP_MAX : v;
        if (counter < COUNTER_MAX)
            counter++;
    }
    return 0;
}

static int dev_on(void)
{
    if (atomic_load(&out_port))
        return 0;
    jack_port_t *p = jack_port_register(client, "out", JACK_DEFAULT_AUDIO_TYPE,
                                        JackPortIsOutput, 0);
    if (!p)
        return -1;
    atomic_store_explicit(&out_port, p, memory_order_release);
    return 0;
}

static void dev_off(void)
{
    jack_port_t *p = atomic_exchange_explicit(&out_port, NULL, memory_order_acq_rel);
    if (p)
        jack_port_unregister(client, p);
}

static void run_for(double seconds)
{
    long slices = (long)(seconds * 20.0);
    for (long i = 0; i < slices && !quit_flag; i++)
        usleep(50000);
}

static unsigned long rng_state = 12345;

static unsigned long xr(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  %s stable    <name> <seconds>\n"
                "  %s disappear <name> <on_s> <off_s> <cycles>\n"
                "  %s flaky     <name> <seconds> <min_ms> <max_ms> [seed]\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    const char *name = argv[2];

    client = jack_client_open(name, JackNullOption, NULL);
    if (!client) {
        fprintf(stderr, "[%s] cannot connect to JACK server\n", name);
        return 1;
    }
    jack_set_process_callback(client, on_process, NULL);
    if (jack_activate(client) != 0) {
        fprintf(stderr, "[%s] cannot activate client\n", name);
        return 1;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    if (!strcmp(mode, "stable") && argc >= 4) {
        dev_on();
        run_for(atof(argv[3]));
    } else if (!strcmp(mode, "disappear") && argc >= 6) {
        double on_s = atof(argv[3]), off_s = atof(argv[4]);
        int cycles = atoi(argv[5]);
        for (int c = 0; c < cycles && !quit_flag; c++) {
            dev_on();
            run_for(on_s);
            dev_off();
            run_for(off_s);
        }
    } else if (!strcmp(mode, "flaky") && argc >= 6) {
        double secs = atof(argv[3]);
        long mn = atol(argv[4]), mx = atol(argv[5]);
        if (argc > 6)
            rng_state = strtoul(argv[6], NULL, 10) | 1ul;
        double elapsed = 0.0;
        dev_on();
        while (elapsed < secs && !quit_flag) {
            long ms = mn + (long)(xr() % (unsigned long)(mx - mn + 1));
            usleep((useconds_t)ms * 1000u);
            elapsed += ms / 1000.0;
            if (atomic_load(&out_port))
                dev_off();
            else
                dev_on();
        }
        dev_on();      /* 结束前恢复在线, 留下一段稳定尾音 */
        run_for(1.0);
    } else {
        fprintf(stderr, "[%s] bad arguments for mode '%s'\n", name, mode);
        jack_client_close(client);
        return 2;
    }

    dev_off();
    jack_client_close(client);
    fprintf(stderr, "[%s] done\n", name);
    return 0;
}
