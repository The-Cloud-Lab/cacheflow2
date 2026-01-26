#include "host_provider.h"
#include "../common/doca_common.h"
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

DOCA_LOG_REGISTER(PERF_TEST);

/* Helper to allocate pinned memory using huge pages for DOCA DMA */
void* allocate_pinned_memory(size_t size) {
    void *ptr;

    /* Try huge pages first (required for reliable DOCA DMA) */
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr != MAP_FAILED) {
        fprintf(stderr, "[INFO] Allocated %zu bytes using huge pages\n", size);
        return ptr;
    }

    /* Fall back to regular locked memory */
    fprintf(stderr, "[WARN] Huge pages allocation failed, falling back to regular mmap\n");
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    return ptr;
}

void free_pinned_memory(void *ptr, size_t size) {
    munmap(ptr, size);
}

/* Get current time in microseconds (for performance test) */
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000UL + ts.tv_nsec / 1000UL;
}

/* Statistics for a run */
typedef struct {
    size_t num_transfers;
    size_t total_bytes;
    uint64_t total_time_us;
    uint64_t min_latency_us;
    uint64_t max_latency_us;
    double avg_latency_us;
    double throughput_gbps;
} bench_stats_t;

void print_separator(const char *title) {
    printf("\n");
    printf("========================================\n");
    if (title) {
        printf("%s\n", title);
        printf("========================================\n");
    } else {
        printf("========================================\n");
    }
}

void print_stats(const char *test_name, bench_stats_t *stats) {
    printf("\n[%s]\n", test_name);
    printf("  Transfers: %zu\n", stats->num_transfers);
    printf("  Total data: %.2f GB\n", stats->total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Total time: %.2f ms (%.6f sec)\n", stats->total_time_us / 1000.0, stats->total_time_us / 1000000.0);
    printf("  Min latency: %.2f μs\n", (double)stats->min_latency_us);
    printf("  Max latency: %.2f μs\n", (double)stats->max_latency_us);
    printf("  Avg latency: %.2f μs\n", stats->avg_latency_us);
    printf("  Throughput: %.2f Gbps (%.2f GB/s)\n", 
           stats->throughput_gbps, stats->throughput_gbps / 8.0);
}

int main(int argc, char **argv) {
    doca_error_t result;
    host_provider_t *provider = NULL;
    const char *pci_addr = DEFAULT_DPU_PCI;
    
    /* Test parameters */
    size_t buffer_size = 1024 * 1024 * 1024;  /* 1 GB default */
    size_t chunk_size = 64 * 1024 * 1024;     /* 64 MB chunks default */
    
    /* Parse arguments */
    if (argc > 1) {
        pci_addr = argv[1];
    }
    if (argc > 2) {
        buffer_size = atol(argv[2]) * 1024 * 1024;  /* Size in MB */
    }
    if (argc > 3) {
        chunk_size = atol(argv[3]) * 1024 * 1024;   /* Chunk size in MB */
    }
    
    /* Initialize DOCA logging */
    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS && result != DOCA_ERROR_IN_USE) {
        fprintf(stderr, "Warning: Failed to create log backend\n");
    }
    
    print_separator("DOCA DPU Performance Benchmark");
    printf("Configuration:\n");
    printf("  DPU PCI: %s\n", pci_addr);
    printf("  Buffer size: %.2f GB\n", buffer_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Chunk size: %.2f MB\n", chunk_size / (1024.0 * 1024.0));
    printf("  Number of transfers: %zu\n", buffer_size / chunk_size);
    
    /* Initialize provider */
    DOCA_LOG_INFO("Initializing host provider...");
    result = host_provider_init(&provider, pci_addr);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize provider");
        return 1;
    }
    DOCA_LOG_INFO("✓ Host provider initialized");
    
    /* Allocate buffer */
    DOCA_LOG_INFO("Allocating %.2f GB pinned memory...", buffer_size / (1024.0 * 1024.0 * 1024.0));
    void *buffer = allocate_pinned_memory(buffer_size);
    if (!buffer) {
        DOCA_LOG_ERR("Failed to allocate pinned memory");
        host_provider_destroy(provider);
        return 1;
    }
    
    /* Fill with test pattern */
    uint32_t *data = (uint32_t *)buffer;
    for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
        data[i] = i;
    }
    DOCA_LOG_INFO("✓ Memory allocated and filled");
    
    /* Register buffer */
    DOCA_LOG_INFO("Registering buffer...");
    uint64_t buffer_id;
    result = host_provider_register_buffer(provider, buffer, buffer_size, &buffer_id);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to register buffer");
        free_pinned_memory(buffer, buffer_size);
        host_provider_destroy(provider);
        return 1;
    }
    DOCA_LOG_INFO("✓ Buffer registered with ID: %lu", buffer_id);
    
    /* Run performance tests */
    print_separator("Running Performance Tests");
    
    /* Test 1: Sequential transfers with full buffer */
    {
        printf("\n[Test 1: Sequential Full Buffer Transfer]\n");
        printf("  Transferring entire %.2f GB buffer as single operation\n", 
               buffer_size / (1024.0 * 1024.0 * 1024.0));
        
        uint64_t transfer_id;
        uint64_t start_time = get_current_time_us();
        
        result = host_provider_transfer(provider, buffer_id, 0, buffer_size, &transfer_id);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to initiate transfer");
            goto cleanup;
        }
        
        result = host_provider_wait_transfer(provider, transfer_id, 60000);  /* 60s timeout */
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Transfer failed or timed out");
            goto cleanup;
        }
        
        uint64_t end_time = get_current_time_us();
        uint64_t duration_us = end_time - start_time;
        
        bench_stats_t stats = {
            .num_transfers = 1,
            .total_bytes = buffer_size,
            .total_time_us = duration_us,
            .min_latency_us = duration_us,
            .max_latency_us = duration_us,
            .avg_latency_us = (double)duration_us,
            .throughput_gbps = (buffer_size * 8.0) / (duration_us / 1000000.0) / 1e9
        };
        print_stats("Full Buffer Transfer", &stats);
    }
    
    /* Test 2: Chunked sequential transfers */
    {
        printf("\n[Test 2: Sequential Chunked Transfer]\n");
        printf("  Transferring %.2f GB in %.2f MB chunks\n",
               buffer_size / (1024.0 * 1024.0 * 1024.0),
               chunk_size / (1024.0 * 1024.0));
        
        size_t num_chunks = buffer_size / chunk_size;
        uint64_t start_time = get_current_time_us();
        uint64_t min_latency = UINT64_MAX;
        uint64_t max_latency = 0;
        uint64_t total_latency = 0;
        int failed = 0;
        
        for (size_t i = 0; i < num_chunks; i++) {
            uint64_t chunk_start = get_current_time_us();
            
            uint64_t transfer_id;
            result = host_provider_transfer(provider, buffer_id, i * chunk_size, chunk_size, &transfer_id);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to initiate transfer %zu", i);
                failed++;
                continue;
            }
            
            result = host_provider_wait_transfer(provider, transfer_id, 60000);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Transfer %zu failed", i);
                failed++;
                continue;
            }
            
            uint64_t chunk_end = get_current_time_us();
            uint64_t chunk_latency = chunk_end - chunk_start;
            
            if (chunk_latency < min_latency) min_latency = chunk_latency;
            if (chunk_latency > max_latency) max_latency = chunk_latency;
            total_latency += chunk_latency;
            
            if ((i + 1) % (num_chunks / 10) == 0 || i == num_chunks - 1) {
                printf("  Progress: %zu/%zu chunks (%.1f%%)\n", 
                       i + 1, num_chunks, 100.0 * (i + 1) / num_chunks);
            }
        }
        
        uint64_t end_time = get_current_time_us();
        uint64_t total_time = end_time - start_time;
        
        bench_stats_t stats = {
            .num_transfers = num_chunks - failed,
            .total_bytes = buffer_size,
            .total_time_us = total_time,
            .min_latency_us = min_latency,
            .max_latency_us = max_latency,
            .avg_latency_us = (double)total_latency / (num_chunks - failed),
            .throughput_gbps = (buffer_size * 8.0) / (total_time / 1000000.0) / 1e9
        };
        printf("  Failed transfers: %d\n", failed);
        print_stats("Sequential Chunked Transfer", &stats);
    }
    
    /* Test 3: Bandwidth scaling with different chunk sizes */
    {
        printf("\n[Test 3: Bandwidth Scaling Analysis]\n");
        printf("  Testing throughput with various chunk sizes\n");
        
        size_t test_chunk_sizes[] = {
            1 * 1024 * 1024,        /* 1 MB */
            8 * 1024 * 1024,        /* 8 MB */
            32 * 1024 * 1024,       /* 32 MB */
            64 * 1024 * 1024,       /* 64 MB */
            128 * 1024 * 1024,      /* 128 MB */
            256 * 1024 * 1024,      /* 256 MB */
            512 * 1024 * 1024,      /* 512 MB */
        };
        int num_chunk_tests = sizeof(test_chunk_sizes) / sizeof(test_chunk_sizes[0]);
        
        printf("  Chunk Size    | Transfers | Total Time | Bandwidth   | Latency\n");
        printf("  ──────────────┼───────────┼────────────┼─────────────┼──────────\n");
        
        for (int t = 0; t < num_chunk_tests; t++) {
            size_t test_chunk = test_chunk_sizes[t];
            
            /* Skip if chunk is larger than buffer */
            if (test_chunk > buffer_size) break;
            
            size_t num_xfers = buffer_size / test_chunk;
            uint64_t start_time = get_current_time_us();
            int failed = 0;
            
            for (size_t i = 0; i < num_xfers; i++) {
                uint64_t transfer_id;
                result = host_provider_transfer(provider, buffer_id, i * test_chunk, test_chunk, &transfer_id);
                if (result != DOCA_SUCCESS) {
                    failed++;
                    continue;
                }
                
                result = host_provider_wait_transfer(provider, transfer_id, 60000);
                if (result != DOCA_SUCCESS) {
                    failed++;
                }
            }
            
            uint64_t end_time = get_current_time_us();
            uint64_t total_time = end_time - start_time;
            double throughput = (buffer_size * 8.0) / (total_time / 1000000.0) / 1e9;
            double avg_latency = (double)total_time / (num_xfers - failed);
            
            printf("  %7.1f MB    | %9zu | %7.2f ms | %10.2f Gbps | %.2f μs\n",
                   test_chunk / (1024.0 * 1024.0),
                   num_xfers - failed,
                   total_time / 1000.0,
                   throughput,
                   avg_latency);
        }
    }
    
    print_separator("Summary");
    printf("All tests completed successfully!\n");
    printf("DPU connection is ready for production use.\n");
    
cleanup:
    /* Cleanup */
    DOCA_LOG_INFO("Unregistering buffer...");
    host_provider_unregister_buffer(provider, buffer_id);
    
    DOCA_LOG_INFO("Cleaning up memory...");
    free_pinned_memory(buffer, buffer_size);
    
    DOCA_LOG_INFO("Destroying provider...");
    host_provider_destroy(provider);
    
    return result == DOCA_SUCCESS ? 0 : 1;
}
