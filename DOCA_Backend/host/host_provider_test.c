#include "host_provider.h"
#include "../common/doca_common.h"
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <errno.h>

DOCA_LOG_REGISTER(HOST_TEST);

/* Helper to allocate pinned memory using huge pages for DOCA DMA */
void* allocate_pinned_memory(size_t size) {
    void *ptr;

    /* Try huge pages first (required for reliable DOCA DMA) */
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr != MAP_FAILED) {
        fprintf(stderr, "[MEMORY] Allocated %zu bytes using HUGE PAGES at %p\n", size, ptr);
        return ptr;
    }
    fprintf(stderr, "[MEMORY] Huge pages failed (errno=%d), trying regular mmap\n", errno);

    /* Fall back to regular locked memory */
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "[MEMORY] Regular mmap also failed (errno=%d)\n", errno);
        return NULL;
    }
    fprintf(stderr, "[MEMORY] Allocated %zu bytes using regular locked memory at %p\n", size, ptr);
    return ptr;
}

void free_pinned_memory(void *ptr, size_t size) {
    munmap(ptr, size);
}

int main(int argc, char **argv) {
    doca_error_t result;
    host_provider_t *provider = NULL;
    const char *pci_addr = DEFAULT_DPU_PCI;
    void *buffer;
    uint32_t *data;
    size_t i;
    uint64_t buffer_id, transfer_id;
    transfer_stats_t stats;
    const size_t test_sizes[] = { 2 * 1024 * 1024, 4 * 1024 * 1024 };
    const uint32_t timeout_ms = 60000;
    
    if (argc > 1) {
        pci_addr = argv[1];
    }
    
    /* Initialize DOCA logging - try to create, but continue if already exists */
    result = doca_log_backend_create_standard();
    if (result == DOCA_SUCCESS) {
        /* Successfully created new log backend */
        fprintf(stderr, "DOCA log backend initialized\n\n");
    } else if (result == DOCA_ERROR_IN_USE) {
        /* Log backend already exists, that's fine */
        fprintf(stderr, "Log backend already initialized (continuing...)\n\n");
    } else {
        /* Real error - but let's try to continue anyway with fprintf */
        fprintf(stderr, "Warning: Failed to create log backend: %s\n", doca_error_get_descr(result));
        fprintf(stderr, "Continuing with basic logging...\n\n");
    }
    
    DOCA_LOG_INFO("========================================");
    DOCA_LOG_INFO("DOCA Host Provider Test");
    DOCA_LOG_INFO("========================================");
    DOCA_LOG_INFO("DPU PCI Address: %s", pci_addr);
    
    /* Initialize provider */
    DOCA_LOG_INFO("Initializing host provider...");
    result = host_provider_init(&provider, pci_addr);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize provider");
        return 1;
    }
    
    DOCA_LOG_INFO("✓ Host provider initialized");
    
    for (size_t t = 0; t < sizeof(test_sizes) / sizeof(test_sizes[0]); t++) {
        size_t buffer_size = test_sizes[t];

        DOCA_LOG_INFO("\n=== Test: Host->DPU and DPU->Host (%zu bytes) ===", buffer_size);

        buffer = allocate_pinned_memory(buffer_size);
        if (!buffer) {
            DOCA_LOG_ERR("Failed to allocate pinned memory");
            host_provider_destroy(provider);
            return 1;
        }

        /* Fill buffer with test pattern */
        DOCA_LOG_INFO("Filling buffer with test pattern...");
        data = (uint32_t *)buffer;
        for (i = 0; i < buffer_size / sizeof(uint32_t); i++) {
            data[i] = (uint32_t)i;
        }

        /* Register buffer */
        DOCA_LOG_INFO("Registering buffer (%zu bytes)...", buffer_size);
        result = host_provider_register_buffer(provider, buffer, buffer_size, &buffer_id);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to register buffer");
            free_pinned_memory(buffer, buffer_size);
            host_provider_destroy(provider);
            return 1;
        }

        DOCA_LOG_INFO("✓ Buffer registered with ID: %lu", buffer_id);

        /* Host -> DPU transfer */
        DOCA_LOG_INFO("Transferring buffer to DPU...");
        result = host_provider_transfer(provider, buffer_id, 0, buffer_size, &transfer_id);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to initiate transfer");
            host_provider_unregister_buffer(provider, buffer_id);
            free_pinned_memory(buffer, buffer_size);
            host_provider_destroy(provider);
            return 1;
        }

        DOCA_LOG_INFO("Transfer initiated with ID: %lu", transfer_id);
        DOCA_LOG_INFO("Waiting for transfer to complete...");
        result = host_provider_wait_transfer(provider, transfer_id, timeout_ms);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Transfer failed or timed out");
            host_provider_unregister_buffer(provider, buffer_id);
            free_pinned_memory(buffer, buffer_size);
            host_provider_destroy(provider);
            return 1;
        }
        DOCA_LOG_INFO("✓ Host->DPU transfer completed successfully");

        /* DPU -> Host load */
        DOCA_LOG_INFO("Loading buffer back to host (DPU->Host)...");
        result = host_provider_load(provider, buffer_id, 0, buffer_size, &transfer_id);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to initiate load");
            host_provider_unregister_buffer(provider, buffer_id);
            free_pinned_memory(buffer, buffer_size);
            host_provider_destroy(provider);
            return 1;
        }

        DOCA_LOG_INFO("Load initiated with ID: %lu", transfer_id);
        DOCA_LOG_INFO("Waiting for load to complete...");
        result = host_provider_wait_transfer(provider, transfer_id, timeout_ms);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Load failed or timed out");
            host_provider_unregister_buffer(provider, buffer_id);
            free_pinned_memory(buffer, buffer_size);
            host_provider_destroy(provider);
            return 1;
        }
        DOCA_LOG_INFO("✓ DPU->Host load completed successfully");

        DOCA_LOG_INFO("Unregistering buffer...");
        host_provider_unregister_buffer(provider, buffer_id);
        free_pinned_memory(buffer, buffer_size);
    }
    
    /* Get statistics */
    DOCA_LOG_INFO("\n=== Transfer Statistics ===");
    result = host_provider_get_stats(provider, &stats);
    if (result == DOCA_SUCCESS) {
        DOCA_LOG_INFO("Total transfers: %lu", stats.total_transfers);
        DOCA_LOG_INFO("Total bytes: %.2f MB", stats.total_bytes / (1024.0 * 1024.0));
        DOCA_LOG_INFO("Failed transfers: %lu", stats.failed_transfers);
        DOCA_LOG_INFO("Avg latency: %.2f μs", stats.avg_latency_us);
        DOCA_LOG_INFO("Peak bandwidth: %.2f Gbps", stats.peak_bandwidth_gbps);
    }
    
    /* Cleanup */
    DOCA_LOG_INFO("\n=== Cleanup ===");
    DOCA_LOG_INFO("Destroying provider...");
    host_provider_destroy(provider);
    
    DOCA_LOG_INFO("\n========================================");
    DOCA_LOG_INFO("✓ All tests passed successfully!");
    DOCA_LOG_INFO("========================================");
    
    return 0;
}
