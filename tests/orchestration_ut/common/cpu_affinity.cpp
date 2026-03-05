/**
 * cpu_affinity.cpp
 *
 * CPU affinity utility function implementation.
 */

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

#include <cstdio>
#include "cpu_affinity.h"

void print_separator(int n) {
    printf("  ");
    for (int i = 0; i < n; i++) putchar('-');
    putchar('\n');
}

int num_cpus_online() {
#ifdef __linux__
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 1;
#endif
}

int current_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

int bind_to_cpu(int cpu_core) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)cpu_core;
    return -1;
#endif
}

int unbind_from_cpu() {
#ifdef __linux__
    int ncpus = num_cpus_online();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < ncpus; i++) CPU_SET(i, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    return -1;
#endif
}

int get_bound_cpu() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        return -1;
    }
    int count = 0;
    int first = -1;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            if (first < 0) first = i;
            count++;
        }
    }
    // If bound to all online cores, consider as "not specifically bound"
    if (count >= num_cpus_online()) return -1;
    return first;
#else
    return -1;
#endif
}

bool verify_cpu_binding(int target_cpu, const char* context) {
#ifdef __linux__
    int bound = get_bound_cpu();
    int actual = current_cpu();
    bool ok = (bound == target_cpu) && (actual == target_cpu);
    if (context && context[0] != '\0') {
        printf("  [%s] Binding verification: expected=%d, bound(read)=%d, actual(measured)=%d => %s\n",
               context, target_cpu, bound, actual, ok ? "OK" : "FAIL");
    } else {
        printf("  Binding verification: expected=%d, bound(read)=%d, actual(measured)=%d => %s\n",
               target_cpu, bound, actual, ok ? "OK" : "FAIL");
    }
    return ok;
#else
    (void)target_cpu;
    (void)context;
    printf("  Binding verification: Non-Linux platform, skipped\n");
    return true;
#endif
}
