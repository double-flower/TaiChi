#ifndef TIMING_H
#define TIMING_H

#include <cuda_runtime.h>
#include <sys/time.h>

struct gpu_timer {
    cudaEvent_t start_event, stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

struct cpu_timer {
    struct timeval start_t, stop_t;

    void start() {
        gettimeofday(&start_t, NULL);
    }

    float stop() {
        gettimeofday(&stop_t, NULL);
        float elapsedTime = ((long)stop_t.tv_sec - (long)start_t.tv_sec) * 1000 
                          + ((long)stop_t.tv_usec - (long)start_t.tv_usec) / 1000;  // millisecond
        return elapsedTime;
    }
};
#endif // TIMING_H
