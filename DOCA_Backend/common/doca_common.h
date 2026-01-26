#ifndef DOCA_COMMON_H
#define DOCA_COMMON_H

#include <doca_error.h>
#include <doca_log.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error checking macro
#define CHECK_DOCA_ERROR(result, msg) \
    do { \
        if ((result) != DOCA_SUCCESS) { \
            DOCA_LOG_ERR("%s: %s", msg, doca_error_get_descr(result)); \
            return result; \
        } \
    } while(0)

// Configuration constants
#define MAX_BUFFER_SIZE (1ULL << 30)  // 1 GB max buffer size
#define MAX_BUFFERS 1024               // Maximum number of buffers
#define DMA_ALIGNMENT 64               // DMA alignment requirement
#define COMM_CHANNEL_TIMEOUT_MS 5000   // Communication channel timeout
#define MAX_QUEUE_DEPTH 512            // Maximum DMA queue depth (increased from 128 to handle burst transfers)

// Device configuration
#define PCI_ADDR_LEN 16
#define DEFAULT_HOST_PCI "0000:01:00.0"  // Adjust for your system
#define DEFAULT_DPU_PCI "0000:0c:00.0"   // BlueField-3 ConnectX-7

// Helper function to align addresses
static inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Helper function to check alignment
static inline int is_aligned(void *ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

// Time measurement helpers
#include <time.h>

static inline double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Bandwidth calculation (bytes/sec -> Gbps)
static inline double bytes_per_sec_to_gbps(double bytes_per_sec) {
    return (bytes_per_sec * 8.0) / 1e9;
}

#ifdef __cplusplus
}
#endif

#endif // DOCA_COMMON_H
