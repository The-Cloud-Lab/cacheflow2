#ifndef HOST_PROVIDER_H
#define HOST_PROVIDER_H

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
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
typedef struct host_buffer host_buffer_t;
typedef struct host_provider host_provider_t;

// Buffer tracking structure
struct host_buffer {
    uint64_t buffer_id;
    void *host_addr;
    size_t size;
    struct doca_mmap *mmap;
    uint8_t *export_desc;
    size_t export_desc_len;
    bool registered;
};

// Host provider context
struct host_provider {
    struct doca_dev *dev;
    struct doca_comch_client *client;
    struct doca_comch_connection *connection;
    struct doca_pe *pe;
    
    // Buffer management
    host_buffer_t *buffers[MAX_BUFFERS];
    uint32_t num_buffers;
    
    // Statistics
    transfer_stats_t stats;
    
    // Configuration
    char pci_addr[PCI_ADDR_LEN];
    char server_name[256];
    bool initialized;
    bool connected;
};

/**
 * Initialize the host provider
 * 
 * @param provider [out] - Pointer to provider structure to initialize
 * @param pci_addr [in] - PCI address of the DPU (e.g., "0000:03:00.0")
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_init(host_provider_t **provider, const char *pci_addr);

/**
 * Register a host memory buffer for DMA access by DPU
 * 
 * @param provider [in] - Host provider context
 * @param host_addr [in] - Host memory address (must be pinned)
 * @param size [in] - Size of the buffer in bytes
 * @param buffer_id [out] - Unique buffer ID assigned
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_register_buffer(
    host_provider_t *provider,
    void *host_addr,
    size_t size,
    uint64_t *buffer_id
);

/**
 * Request a DMA transfer from host to DPU
 * 
 * @param provider [in] - Host provider context
 * @param buffer_id [in] - Buffer ID to transfer
 * @param offset [in] - Offset within buffer
 * @param length [in] - Number of bytes to transfer
 * @param transfer_id [out] - Transfer ID for tracking
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_transfer(
    host_provider_t *provider,
    uint64_t buffer_id,
    uint64_t offset,
    size_t length,
    uint64_t *transfer_id
);

/**
 * Request a DMA load from DPU to host (reverse direction)
 *
 * @param provider [in] - Host provider context
 * @param buffer_id [in] - Buffer ID to load into
 * @param offset [in] - Offset within buffer
 * @param length [in] - Number of bytes to load
 * @param transfer_id [out] - Transfer ID for tracking
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_load(
    host_provider_t *provider,
    uint64_t buffer_id,
    uint64_t offset,
    size_t length,
    uint64_t *transfer_id
);

/**
 * Wait for a transfer to complete
 *
 * @param provider [in] - Host provider context
 * @param transfer_id [in] - Transfer ID to wait for
 * @param timeout_ms [in] - Timeout in milliseconds (0 = no timeout)
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_wait_transfer(
    host_provider_t *provider,
    uint64_t transfer_id,
    uint32_t timeout_ms
);

/**
 * Unregister a buffer
 * 
 * @param provider [in] - Host provider context
 * @param buffer_id [in] - Buffer ID to unregister
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_unregister_buffer(
    host_provider_t *provider,
    uint64_t buffer_id
);

/**
 * Get transfer statistics
 * 
 * @param provider [in] - Host provider context
 * @param stats [out] - Statistics structure to fill
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t host_provider_get_stats(
    host_provider_t *provider,
    transfer_stats_t *stats
);

/**
 * Destroy the host provider and cleanup resources
 * 
 * @param provider [in] - Host provider context to destroy
 */
void host_provider_destroy(host_provider_t *provider);

#ifdef __cplusplus
}
#endif

#endif // HOST_PROVIDER_H
