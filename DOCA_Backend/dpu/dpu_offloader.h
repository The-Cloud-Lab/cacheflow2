#ifndef DPU_OFFLOADER_H
#define DPU_OFFLOADER_H

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_dma.h>
#include <doca_mmap.h>
#include <doca_comch.h>
#include <doca_pe.h>
#include <stdint.h>
#include <stdbool.h>
#include "../common/doca_common.h"
#include "../common/protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct dpu_buffer dpu_buffer_t;
typedef struct dpu_offloader dpu_offloader_t;

// DPU-side buffer structure
struct dpu_buffer {
    uint64_t buffer_id;
    uint64_t host_addr;
    size_t size;
    struct doca_mmap *remote_mmap;  // Imported host memory
    struct doca_mmap *local_mmap;   // Local DPU memory
    void *local_addr;               // Local buffer address
    bool local_addr_is_mmap;        // True if local_addr came from mmap()
    bool registered;
};

// Forward declare comch_cfg for the header
struct comch_cfg;

// DPU offloader context
struct dpu_offloader {
    struct doca_dev *dev;
    struct doca_dev_rep *dev_rep;
    struct doca_dma *dma;
    struct doca_pe *pe;
    struct doca_comch_server *server;
    struct doca_comch_connection *connection;
    struct comch_cfg *comch_cfg;  // Communication channel config for reconnection handling

    struct doca_buf_inventory *buf_inventory;

    // Buffer management
    dpu_buffer_t *buffers[MAX_BUFFERS];
    uint32_t num_buffers;

    // DMA capabilities
    uint64_t max_dma_buf_size;  // Maximum buffer size per DMA operation

    // Statistics
    transfer_stats_t stats;

    // Configuration
    char server_name[256];
    bool initialized;
    bool running;
};

/**
 * Initialize the DPU offloader
 * 
 * @param offloader [out] - Pointer to offloader structure to initialize
 * @param server_addr [in] - Host server address to connect to (IP:port), NULL for standalone mode
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t dpu_offloader_init(dpu_offloader_t **offloader, const char *server_addr);

/**
 * Initialize the DPU offloader in standalone mode (no ComCh, DMA testing only)
 * 
 * @param offloader [out] - Pointer to offloader structure to initialize
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t dpu_offloader_init_standalone(dpu_offloader_t **offloader);

/**
 * Start the DPU offloader service
 * This function will run the main loop and process transfer requests
 * 
 * @param offloader [in] - DPU offloader context
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t dpu_offloader_run(dpu_offloader_t *offloader);

/**
 * Stop the DPU offloader service
 * 
 * @param offloader [in] - DPU offloader context
 */
void dpu_offloader_stop(dpu_offloader_t *offloader);

/**
 * Get transfer statistics
 * 
 * @param offloader [in] - DPU offloader context
 * @param stats [out] - Statistics structure to fill
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t dpu_offloader_get_stats(
    dpu_offloader_t *offloader,
    transfer_stats_t *stats
);

/**
 * Destroy the DPU offloader and cleanup resources
 * 
 * @param offloader [in] - DPU offloader context to destroy
 */
void dpu_offloader_destroy(dpu_offloader_t *offloader);

#ifdef __cplusplus
}
#endif

#endif // DPU_OFFLOADER_H
