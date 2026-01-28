/*
 * DOCA_Backend/dpu/dpu_offloader.c
 * Fixes: Compilation errors (missing headers, symbol conflicts)
 */
#include "dpu_offloader.h"
#include "../common/doca_common.h"
#include "../common/comch_utils.h"
#include "../common/common.h"  /* Added for open_doca_device_with_pci */
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/mman.h>

DOCA_LOG_REGISTER(DPU_OFFLOADER);

static volatile bool keep_running = true;

typedef struct {
    uint64_t transfer_id;
    struct doca_comch_connection *connection;
    struct doca_buf *src_buf;
    struct doca_buf *dst_buf;
    dpu_offloader_t *off;
    size_t length;
    double start_time;
    /* Chunked transfer tracking */
    size_t total_length;        /* Total transfer length */
    size_t chunks_total;        /* Total number of chunks */
    size_t chunks_completed;    /* Chunks successfully completed */
    size_t chunks_failed;       /* Chunks that failed */
    bool is_chunked;            /* Whether this is part of a chunked transfer */
    void *chunk_tracker;        /* Shared tracker for chunked transfers */
} offload_ctx_t;

/* Tracker for chunked transfers - shared across all chunks of a single transfer */
typedef struct {
    uint64_t transfer_id;
    struct doca_comch_connection *connection;
    dpu_offloader_t *off;
    size_t total_length;
    size_t chunks_total;
    volatile size_t chunks_completed;
    volatile size_t chunks_failed;
    double start_time;
} chunk_tracker_t;

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        DOCA_LOG_INFO("Received signal %d, shutting down...", signum);
        keep_running = false;
    }
}

/* Helper to destroy a single buffer safely */
static void destroy_dpu_buffer(dpu_buffer_t *buf) {
    if (!buf) return;
    if (buf->local_mmap) doca_mmap_destroy(buf->local_mmap);
    if (buf->remote_mmap) doca_mmap_destroy(buf->remote_mmap);
    if (buf->local_addr) {
        if (buf->local_addr_is_mmap) {
            munmap(buf->local_addr, buf->size);
        } else {
            free(buf->local_addr);
        }
    }
    free(buf);
}

/* Scans for a device that supports DMA Memory Copy 
 * This avoids hardcoding "03:00.0" which fails on some BF3 configurations
 */
static doca_error_t open_dma_device(struct doca_dev **dev) {
    struct doca_devinfo **dev_list;
    uint32_t nb_devs;
    doca_error_t result;
    uint32_t i;
    
    result = doca_devinfo_create_list(&dev_list, &nb_devs);
    CHECK_DOCA_ERROR(result, "Failed to create device list");
    
    for (i = 0; i < nb_devs; i++) {
        result = doca_dma_cap_task_memcpy_is_supported(dev_list[i]);
        if (result == DOCA_SUCCESS) {
            result = doca_dev_open(dev_list[i], dev);
            if (result == DOCA_SUCCESS) {
                DOCA_LOG_INFO("Selected DMA device index: %d", i);
            }
            doca_devinfo_destroy_list(dev_list);
            return result;
        }
    }
    doca_devinfo_destroy_list(dev_list);
    return DOCA_ERROR_NOT_FOUND;
}

/* Renamed to avoid conflict with common.c's open_doca_device_rep_with_pci */
static doca_error_t open_representor_custom(struct doca_dev *dev, const char *rep_pci_addr, struct doca_dev_rep **dev_rep) {
    struct doca_devinfo_rep **rep_list;
    uint32_t nb_reps, i;
    doca_error_t result;
    uint8_t is_addr_equal;
    
    /* Try Net filter first, then All */
    result = doca_devinfo_rep_create_list(dev, DOCA_DEVINFO_REP_FILTER_NET, &rep_list, &nb_reps);
    if (result != DOCA_SUCCESS) {
        result = doca_devinfo_rep_create_list(dev, DOCA_DEVINFO_REP_FILTER_ALL, &rep_list, &nb_reps);
        if (result != DOCA_SUCCESS) return result;
    }
    
    /* If specific PCI requested */
    if (rep_pci_addr && strlen(rep_pci_addr) > 0) {
        for (i = 0; i < nb_reps; i++) {
            result = doca_devinfo_rep_is_equal_pci_addr(rep_list[i], rep_pci_addr, &is_addr_equal);
            if (result == DOCA_SUCCESS && is_addr_equal) {
                result = doca_dev_rep_open(rep_list[i], dev_rep);
                if (result == DOCA_SUCCESS) {
                    doca_devinfo_rep_destroy_list(rep_list);
                    return DOCA_SUCCESS;
                }
            }
        }
    }
    
    /* Fallback: Open first available representor if specific one not found or not requested */
    if (nb_reps > 0) {
        result = doca_dev_rep_open(rep_list[0], dev_rep);
        if (result == DOCA_SUCCESS) {
            doca_devinfo_rep_destroy_list(rep_list);
            return DOCA_SUCCESS;
        }
    }
    doca_devinfo_rep_destroy_list(rep_list);
    return DOCA_ERROR_NOT_FOUND;
}

static void send_transfer_completion(struct doca_comch_connection *connection, uint64_t transfer_id,
                                      error_code_t status, size_t bytes_transferred) {
    doca_message_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.type = MSG_TRANSFER_COMPLETE;
    msg.header.version = PROTOCOL_VERSION;
    msg.header.sequence = transfer_id;
    msg.header.payload_size = sizeof(transfer_complete_msg_t);
    msg.payload.transfer_comp.transfer_id = transfer_id;
    msg.payload.transfer_comp.status = status;
    msg.payload.transfer_comp.bytes_transferred = bytes_transferred;
    comch_utils_send(connection, &msg, sizeof(message_header_t) + msg.header.payload_size);
}

static void dma_completed_callback(struct doca_dma_task_memcpy *task, union doca_data task_user_data, union doca_data ctx_user_data) {
    offload_ctx_t *ctx = (offload_ctx_t *)task_user_data.ptr;
    (void)ctx_user_data;

    /* Release buffers */
    doca_buf_dec_refcount(ctx->src_buf, NULL);
    doca_buf_dec_refcount(ctx->dst_buf, NULL);
    doca_task_free(doca_dma_task_memcpy_as_task(task));

    if (ctx->is_chunked && ctx->chunk_tracker) {
        /* Chunked transfer - update tracker */
        chunk_tracker_t *tracker = (chunk_tracker_t *)ctx->chunk_tracker;
        tracker->chunks_completed++;

        DOCA_LOG_DBG("Chunk completed for transfer %lu: %zu/%zu",
                     tracker->transfer_id, tracker->chunks_completed, tracker->chunks_total);

        /* Check if all chunks are done */
        if (tracker->chunks_completed + tracker->chunks_failed >= tracker->chunks_total) {
            double latency = get_time_us() - tracker->start_time;
            double bandwidth = bytes_per_sec_to_gbps(tracker->total_length / (latency / 1e6));

            tracker->off->stats.total_transfers++;
            tracker->off->stats.total_bytes += tracker->total_length;

            if (tracker->off->stats.avg_latency_us == 0)
                tracker->off->stats.avg_latency_us = latency;
            else
                tracker->off->stats.avg_latency_us = (tracker->off->stats.avg_latency_us * 0.9) + (latency * 0.1);

            if (bandwidth > tracker->off->stats.peak_bandwidth_gbps)
                tracker->off->stats.peak_bandwidth_gbps = bandwidth;

            error_code_t status = (tracker->chunks_failed == 0) ? ERR_SUCCESS : ERR_DMA_FAILED;
            send_transfer_completion(tracker->connection, tracker->transfer_id, status, tracker->total_length);

            DOCA_LOG_INFO("Chunked transfer %lu completed: %zu chunks, %.2f MB, %.2f ms, %.2f Gbps",
                          tracker->transfer_id, tracker->chunks_total,
                          tracker->total_length / (1024.0 * 1024.0), latency / 1000.0, bandwidth);

            free(tracker);
        }
    } else {
        /* Single transfer - send completion immediately */
        double latency = get_time_us() - ctx->start_time;
        double bandwidth = bytes_per_sec_to_gbps(ctx->length / (latency / 1e6));

        ctx->off->stats.total_transfers++;
        ctx->off->stats.total_bytes += ctx->length;

        if (ctx->off->stats.avg_latency_us == 0)
            ctx->off->stats.avg_latency_us = latency;
        else
            ctx->off->stats.avg_latency_us = (ctx->off->stats.avg_latency_us * 0.9) + (latency * 0.1);

        if (bandwidth > ctx->off->stats.peak_bandwidth_gbps)
            ctx->off->stats.peak_bandwidth_gbps = bandwidth;

        send_transfer_completion(ctx->connection, ctx->transfer_id, ERR_SUCCESS, ctx->length);
    }

    free(ctx);
}

static void dma_error_callback(struct doca_dma_task_memcpy *task, union doca_data task_user_data, union doca_data ctx_user_data) {
    offload_ctx_t *ctx = (offload_ctx_t *)task_user_data.ptr;
    struct doca_task *doca_task = doca_dma_task_memcpy_as_task(task);
    doca_error_t error_result;
    (void)ctx_user_data;

    error_result = doca_task_get_status(doca_task);
    DOCA_LOG_ERR("DMA transfer %lu failed: %s (%d)", ctx->transfer_id, doca_error_get_descr(error_result), error_result);

    /* Release buffers */
    doca_buf_dec_refcount(ctx->src_buf, NULL);
    doca_buf_dec_refcount(ctx->dst_buf, NULL);
    doca_task_free(doca_task);

    if (ctx->is_chunked && ctx->chunk_tracker) {
        /* Chunked transfer - update tracker */
        chunk_tracker_t *tracker = (chunk_tracker_t *)ctx->chunk_tracker;
        tracker->chunks_failed++;

        DOCA_LOG_ERR("Chunk failed for transfer %lu: %zu/%zu (failed: %zu)",
                     tracker->transfer_id, tracker->chunks_completed + tracker->chunks_failed,
                     tracker->chunks_total, tracker->chunks_failed);

        /* Check if all chunks are done */
        if (tracker->chunks_completed + tracker->chunks_failed >= tracker->chunks_total) {
            tracker->off->stats.failed_transfers++;
            send_transfer_completion(tracker->connection, tracker->transfer_id, ERR_DMA_FAILED, 0);
            free(tracker);
        }
    } else {
        /* Single transfer - send error immediately */
        ctx->off->stats.failed_transfers++;
        send_transfer_completion(ctx->connection, ctx->transfer_id, ERR_DMA_FAILED, 0);
    }

    free(ctx);
}

static doca_error_t handle_buffer_registration(dpu_offloader_t *off, buffer_register_msg_t *reg, struct doca_comch_connection *connection) {
    doca_error_t result;
    dpu_buffer_t *buf;
    doca_message_t ack_msg;

    DOCA_LOG_INFO("Registering buffer %lu (size=%zu, host_addr=0x%lx, export_desc_len=%zu)",
                  reg->buffer_id, reg->size, reg->host_addr, reg->export_desc_len);

    if (reg->buffer_id >= MAX_BUFFERS) {
        DOCA_LOG_ERR("Buffer ID %lu exceeds MAX_BUFFERS %d", reg->buffer_id, MAX_BUFFERS);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (reg->export_desc_len == 0 || reg->export_desc_len > sizeof(reg->export_desc)) {
        DOCA_LOG_ERR("Invalid export descriptor length: %zu (max=%zu)",
                     reg->export_desc_len, sizeof(reg->export_desc));
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (off->buffers[reg->buffer_id] != NULL) {
        DOCA_LOG_INFO("Replacing existing buffer %lu", reg->buffer_id);
        destroy_dpu_buffer(off->buffers[reg->buffer_id]);
        off->buffers[reg->buffer_id] = NULL;
    }

    buf = (dpu_buffer_t *)calloc(1, sizeof(dpu_buffer_t));
    buf->buffer_id = reg->buffer_id;
    buf->host_addr = reg->host_addr;
    buf->size = reg->size;

    /* Import remote memory - this creates an mmap from the host's exported descriptor */
    DOCA_LOG_DBG("Importing mmap from export descriptor (%zu bytes)", reg->export_desc_len);
    result = doca_mmap_create_from_export(NULL, (const void *)reg->export_desc, reg->export_desc_len, off->dev, &buf->remote_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to import mmap from export: %s (export_desc_len=%zu)",
                     doca_error_get_descr(result), reg->export_desc_len);
        free(buf);
        return result;
    }
    DOCA_LOG_INFO("Successfully imported remote mmap for buffer %lu", reg->buffer_id);
    
    #ifndef MAP_HUGE_2MB
    #define MAP_HUGE_2MB (21 << 26)
    #endif

    /* Allocate local DPU buffer - Explicitly request 2MB pages */
    buf->local_addr = mmap(NULL, reg->size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_POPULATE, -1, 0);

    if (buf->local_addr == MAP_FAILED) {
        DOCA_LOG_WARN("Huge page allocation failed, falling back to aligned alloc");
        buf->local_addr = aligned_alloc(DMA_ALIGNMENT, align_up(reg->size, DMA_ALIGNMENT));
        if (buf->local_addr == NULL) {
            DOCA_LOG_ERR("Failed to allocate local buffer of size %zu", reg->size);
            doca_mmap_destroy(buf->remote_mmap);
            free(buf);
            return DOCA_ERROR_NO_MEMORY;
        }
        buf->local_addr_is_mmap = false;
    } else {
        buf->local_addr_is_mmap = true;
    }

    DOCA_LOG_INFO("Allocated local buffer at %p (size=%zu)", buf->local_addr, reg->size);

    result = doca_mmap_create(&buf->local_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create local mmap: %s", doca_error_get_descr(result));
        free(buf->local_addr);
        doca_mmap_destroy(buf->remote_mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_add_dev(buf->local_mmap, off->dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add device to local mmap: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->local_mmap);
        free(buf->local_addr);
        doca_mmap_destroy(buf->remote_mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_set_memrange(buf->local_mmap, buf->local_addr, reg->size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set memrange on local mmap: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->local_mmap);
        free(buf->local_addr);
        doca_mmap_destroy(buf->remote_mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_set_permissions(buf->local_mmap,
                                       DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                       DOCA_ACCESS_FLAG_PCI_READ_WRITE |
                                       DOCA_ACCESS_FLAG_RDMA_READ |
                                       DOCA_ACCESS_FLAG_RDMA_WRITE);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set permissions on local mmap: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->local_mmap);
        free(buf->local_addr);
        doca_mmap_destroy(buf->remote_mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_start(buf->local_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start local mmap: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->local_mmap);
        free(buf->local_addr);
        doca_mmap_destroy(buf->remote_mmap);
        free(buf);
        return result;
    }

    DOCA_LOG_INFO("Local mmap started successfully for buffer %lu", reg->buffer_id);
    
    buf->registered = true;
    off->buffers[reg->buffer_id] = buf;
    if (reg->buffer_id >= off->num_buffers) off->num_buffers = reg->buffer_id + 1;
    
    memset(&ack_msg, 0, sizeof(ack_msg));
    ack_msg.header.type = MSG_ACK;
    ack_msg.header.sequence = reg->buffer_id;
    comch_utils_send(connection, &ack_msg, sizeof(message_header_t));
    
    return DOCA_SUCCESS;
}

/* Submit a single DMA chunk */
static doca_error_t submit_dma_chunk(dpu_offloader_t *off, dpu_buffer_t *buf,
                                      uint64_t offset, size_t length,
                                      uint64_t transfer_id, struct doca_comch_connection *connection,
                                      bool is_chunked, chunk_tracker_t *tracker, double start_time) {
    doca_error_t result;
    struct doca_buf *src_buf = NULL, *dst_buf = NULL;
    struct doca_dma_task_memcpy *task = NULL;
    offload_ctx_t *ctx = NULL;
    union doca_data user_data;

    void *src_addr = (void *)(buf->host_addr + offset);
    void *dst_addr = (void *)((uint64_t)buf->local_addr + offset);

    /* Create source buffer */
    result = doca_buf_inventory_buf_get_by_addr(off->buf_inventory, buf->remote_mmap,
                                                 src_addr, length, &src_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create source buffer: %s", doca_error_get_descr(result));
        return result;
    }

    /* Set data position and length for the source buffer */
    result = doca_buf_set_data(src_buf, src_addr, length);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set data for source buffer: %s", doca_error_get_descr(result));
        doca_buf_dec_refcount(src_buf, NULL);
        return result;
    }

    /* Create destination buffer */
    result = doca_buf_inventory_buf_get_by_addr(off->buf_inventory, buf->local_mmap,
                                                 dst_addr, length, &dst_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create dest buffer: %s", doca_error_get_descr(result));
        doca_buf_dec_refcount(src_buf, NULL);
        return result;
    }

    /* Create context for this chunk */
    ctx = calloc(1, sizeof(offload_ctx_t));
    ctx->transfer_id = transfer_id;
    ctx->connection = connection;
    ctx->src_buf = src_buf;
    ctx->dst_buf = dst_buf;
    ctx->off = off;
    ctx->length = length;
    ctx->start_time = start_time;
    ctx->is_chunked = is_chunked;
    ctx->chunk_tracker = tracker;
    user_data.ptr = ctx;

    /* Allocate and submit DMA task */
    result = doca_dma_task_memcpy_alloc_init(off->dma, src_buf, dst_buf, user_data, &task);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to allocate DMA task: %s", doca_error_get_descr(result));
        doca_buf_dec_refcount(src_buf, NULL);
        doca_buf_dec_refcount(dst_buf, NULL);
        free(ctx);
        return result;
    }

    result = doca_task_submit(doca_dma_task_memcpy_as_task(task));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to submit DMA task: %s", doca_error_get_descr(result));
        doca_task_free(doca_dma_task_memcpy_as_task(task));
        doca_buf_dec_refcount(src_buf, NULL);
        doca_buf_dec_refcount(dst_buf, NULL);
        free(ctx);
        return result;
    }

    return DOCA_SUCCESS;
}

static doca_error_t handle_transfer_request(dpu_offloader_t *off, transfer_request_msg_t *req, struct doca_comch_connection *connection) {
    doca_error_t result;
    dpu_buffer_t *buf;

    DOCA_LOG_INFO("Processing transfer %lu: buffer_id=%lu, offset=%lu, length=%zu",
                  req->transfer_id, req->buffer_id, req->offset, req->length);

    if (req->buffer_id >= MAX_BUFFERS || !off->buffers[req->buffer_id]) {
        DOCA_LOG_ERR("Invalid buffer ID %lu (max=%d)", req->buffer_id, MAX_BUFFERS);
        return DOCA_ERROR_INVALID_VALUE;
    }
    buf = off->buffers[req->buffer_id];

    /* Validate offset and length are within buffer bounds */
    if (req->offset + req->length > buf->size) {
        DOCA_LOG_ERR("Transfer request out of bounds: offset=%lu + length=%zu > size=%zu",
                     req->offset, req->length, buf->size);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (req->length == 0) {
        DOCA_LOG_ERR("Transfer length is zero");
        return DOCA_ERROR_INVALID_VALUE;
    }

    double start_time = get_time_us();

    /* Check if we need to chunk this transfer */
    if (req->length <= off->max_dma_buf_size) {
        /* Single transfer - no chunking needed */
        DOCA_LOG_DBG("Single DMA transfer: %zu bytes", req->length);
        return submit_dma_chunk(off, buf, req->offset, req->length,
                                req->transfer_id, connection, false, NULL, start_time);
    }

    /* Chunked transfer needed */
    size_t num_chunks = (req->length + off->max_dma_buf_size - 1) / off->max_dma_buf_size;
    DOCA_LOG_INFO("Chunked transfer: %zu bytes in %zu chunks (max chunk: %lu bytes)",
                  req->length, num_chunks, off->max_dma_buf_size);

    /* Create tracker for this chunked transfer */
    chunk_tracker_t *tracker = calloc(1, sizeof(chunk_tracker_t));
    tracker->transfer_id = req->transfer_id;
    tracker->connection = connection;
    tracker->off = off;
    tracker->total_length = req->length;
    tracker->chunks_total = num_chunks;
    tracker->chunks_completed = 0;
    tracker->chunks_failed = 0;
    tracker->start_time = start_time;

    /* Submit all chunks */
    size_t remaining = req->length;
    uint64_t current_offset = req->offset;

    for (size_t i = 0; i < num_chunks; i++) {
        size_t chunk_size = (remaining > off->max_dma_buf_size) ? off->max_dma_buf_size : remaining;

        result = submit_dma_chunk(off, buf, current_offset, chunk_size,
                                   req->transfer_id, connection, true, tracker, start_time);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to submit chunk %zu/%zu: %s", i + 1, num_chunks, doca_error_get_descr(result));
            tracker->chunks_failed++;
            /* Continue trying to submit remaining chunks */
        }

        current_offset += chunk_size;
        remaining -= chunk_size;
    }

    /* If all chunks failed to submit, clean up and report error */
    if (tracker->chunks_failed == tracker->chunks_total) {
        DOCA_LOG_ERR("All chunks failed to submit for transfer %lu", req->transfer_id);
        send_transfer_completion(connection, req->transfer_id, ERR_DMA_FAILED, 0);
        free(tracker);
        return DOCA_ERROR_IO_FAILED;
    }

    return DOCA_SUCCESS;
}

/* Submit a single DMA load chunk (DPU -> Host, reverse direction) */
static doca_error_t submit_dma_load_chunk(dpu_offloader_t *off, dpu_buffer_t *buf,
                                          uint64_t offset, size_t length,
                                          uint64_t transfer_id, struct doca_comch_connection *connection,
                                          bool is_chunked, chunk_tracker_t *tracker, double start_time) {
    doca_error_t result;
    struct doca_buf *src_buf = NULL, *dst_buf = NULL;
    struct doca_dma_task_memcpy *task = NULL;
    offload_ctx_t *ctx = NULL;
    union doca_data user_data;

    /* REVERSED: Source is DPU local memory, Destination is Host remote memory */
    void *src_addr = (void *)((uint64_t)buf->local_addr + offset);  /* DPU local */
    void *dst_addr = (void *)(buf->host_addr + offset);              /* Host remote */

    /* Create source buffer from LOCAL mmap (DPU memory) */
    result = doca_buf_inventory_buf_get_by_addr(off->buf_inventory, buf->local_mmap,
                                                 src_addr, length, &src_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("LOAD: Failed to create source buffer from local mmap: %s", doca_error_get_descr(result));
        return result;
    }

    /* Set data position and length for the source buffer */
    result = doca_buf_set_data(src_buf, src_addr, length);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("LOAD: Failed to set data for source buffer: %s", doca_error_get_descr(result));
        doca_buf_dec_refcount(src_buf, NULL);
        return result;
    }

    /* Create destination buffer from REMOTE mmap (Host memory) */
    result = doca_buf_inventory_buf_get_by_addr(off->buf_inventory, buf->remote_mmap,
                                                 dst_addr, length, &dst_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("LOAD: Failed to create dest buffer from remote mmap: %s", doca_error_get_descr(result));
        doca_buf_dec_refcount(src_buf, NULL);
        return result;
    }

    /* Create context for this chunk */
    ctx = calloc(1, sizeof(offload_ctx_t));
    ctx->transfer_id = transfer_id;
    ctx->connection = connection;
    ctx->src_buf = src_buf;
    ctx->dst_buf = dst_buf;
    ctx->off = off;
    ctx->length = length;
    ctx->start_time = start_time;
    ctx->is_chunked = is_chunked;
    ctx->chunk_tracker = tracker;
    user_data.ptr = ctx;

    /* Allocate and submit DMA task (DPU local -> Host remote) */
    result = doca_dma_task_memcpy_alloc_init(off->dma, src_buf, dst_buf, user_data, &task);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("LOAD: Failed to allocate DMA task: %s", doca_error_get_descr(result));
        doca_buf_dec_refcount(src_buf, NULL);
        doca_buf_dec_refcount(dst_buf, NULL);
        free(ctx);
        return result;
    }

    result = doca_task_submit(doca_dma_task_memcpy_as_task(task));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("LOAD: Failed to submit DMA task: %s", doca_error_get_descr(result));
        doca_task_free(doca_dma_task_memcpy_as_task(task));
        doca_buf_dec_refcount(src_buf, NULL);
        doca_buf_dec_refcount(dst_buf, NULL);
        free(ctx);
        return result;
    }

    DOCA_LOG_DBG("LOAD: Submitted DMA chunk: DPU %p -> Host %p, %zu bytes", src_addr, dst_addr, length);
    return DOCA_SUCCESS;
}

/* Handle load request (DPU -> Host transfer) */
static doca_error_t handle_load_request(dpu_offloader_t *off, load_request_msg_t *req, struct doca_comch_connection *connection) {
    doca_error_t result;
    dpu_buffer_t *buf;

    DOCA_LOG_INFO("Processing LOAD %lu: buffer_id=%lu, offset=%lu, length=%zu",
                  req->transfer_id, req->buffer_id, req->offset, req->length);

    if (req->buffer_id >= MAX_BUFFERS || !off->buffers[req->buffer_id]) {
        DOCA_LOG_ERR("LOAD: Invalid buffer ID %lu (max=%d)", req->buffer_id, MAX_BUFFERS);
        send_transfer_completion(connection, req->transfer_id, ERR_INVALID_BUFFER, 0);
        return DOCA_ERROR_INVALID_VALUE;
    }
    buf = off->buffers[req->buffer_id];

    /* Validate offset and length are within buffer bounds */
    if (req->offset + req->length > buf->size) {
        DOCA_LOG_ERR("LOAD: Request out of bounds: offset=%lu + length=%zu > size=%zu",
                     req->offset, req->length, buf->size);
        send_transfer_completion(connection, req->transfer_id, ERR_INVALID_BUFFER, 0);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (req->length == 0) {
        DOCA_LOG_ERR("LOAD: Transfer length is zero");
        send_transfer_completion(connection, req->transfer_id, ERR_INVALID_BUFFER, 0);
        return DOCA_ERROR_INVALID_VALUE;
    }

    double start_time = get_time_us();

    /* Check if we need to chunk this transfer */
    if (req->length <= off->max_dma_buf_size) {
        /* Single transfer - no chunking needed */
        DOCA_LOG_DBG("LOAD: Single DMA transfer: %zu bytes", req->length);
        return submit_dma_load_chunk(off, buf, req->offset, req->length,
                                     req->transfer_id, connection, false, NULL, start_time);
    }

    /* Chunked transfer needed */
    size_t num_chunks = (req->length + off->max_dma_buf_size - 1) / off->max_dma_buf_size;
    DOCA_LOG_INFO("LOAD: Chunked transfer: %zu bytes in %zu chunks (max chunk: %lu bytes)",
                  req->length, num_chunks, off->max_dma_buf_size);

    /* Create tracker for this chunked transfer */
    chunk_tracker_t *tracker = calloc(1, sizeof(chunk_tracker_t));
    tracker->transfer_id = req->transfer_id;
    tracker->connection = connection;
    tracker->off = off;
    tracker->total_length = req->length;
    tracker->chunks_total = num_chunks;
    tracker->chunks_completed = 0;
    tracker->chunks_failed = 0;
    tracker->start_time = start_time;

    /* Submit all chunks */
    size_t remaining = req->length;
    uint64_t current_offset = req->offset;

    for (size_t i = 0; i < num_chunks; i++) {
        size_t chunk_size = (remaining > off->max_dma_buf_size) ? off->max_dma_buf_size : remaining;

        result = submit_dma_load_chunk(off, buf, current_offset, chunk_size,
                                       req->transfer_id, connection, true, tracker, start_time);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("LOAD: Failed to submit chunk %zu/%zu: %s", i + 1, num_chunks, doca_error_get_descr(result));
            tracker->chunks_failed++;
        }

        current_offset += chunk_size;
        remaining -= chunk_size;
    }

    /* If all chunks failed to submit, clean up and report error */
    if (tracker->chunks_failed == tracker->chunks_total) {
        DOCA_LOG_ERR("LOAD: All chunks failed to submit for transfer %lu", req->transfer_id);
        send_transfer_completion(connection, req->transfer_id, ERR_DMA_FAILED, 0);
        free(tracker);
        return DOCA_ERROR_IO_FAILED;
    }

    return DOCA_SUCCESS;
}

static void handle_buffer_unregister(dpu_offloader_t *off, buffer_unregister_msg_t *unreg, struct doca_comch_connection *connection) {
    uint64_t buffer_id = unreg->buffer_id;
    doca_message_t ack_msg;
    
    DOCA_LOG_INFO("Unregistering buffer %lu", buffer_id);
    
    if (buffer_id >= MAX_BUFFERS) {
        DOCA_LOG_ERR("Buffer ID %lu exceeds MAX_BUFFERS %d", buffer_id, MAX_BUFFERS);
        return;
    }
    
    if (off->buffers[buffer_id] != NULL) {
        destroy_dpu_buffer(off->buffers[buffer_id]);
        off->buffers[buffer_id] = NULL;
        DOCA_LOG_INFO("Successfully unregistered buffer %lu", buffer_id);
    } else {
        DOCA_LOG_WARN("Buffer %lu not found (already unregistered?)", buffer_id);
    }
    
    /* Send ACK */
    memset(&ack_msg, 0, sizeof(ack_msg));
    ack_msg.header.type = MSG_ACK;
    ack_msg.header.sequence = buffer_id;
    ack_msg.header.payload_size = 0;
    comch_utils_send(connection, &ack_msg, sizeof(message_header_t));
}

static void server_recv_callback(struct doca_comch_event_msg_recv *event, uint8_t *msg, uint32_t msg_len, struct doca_comch_connection *connection) {
    dpu_offloader_t *off = (dpu_offloader_t *)comch_utils_get_user_data(connection);
    doca_message_t *received_msg = (doca_message_t *)msg;
    (void)event;

    if (msg_len < sizeof(message_header_t)) return;

    switch (received_msg->header.type) {
        case MSG_REGISTER_BUFFER:
            handle_buffer_registration(off, &received_msg->payload.buffer_reg, connection);
            break;
        case MSG_TRANSFER_REQUEST:
            handle_transfer_request(off, &received_msg->payload.transfer_req, connection);
            break;
        case MSG_LOAD_REQUEST:
            handle_load_request(off, &received_msg->payload.load_req, connection);
            break;
        case MSG_UNREGISTER_BUFFER:
            handle_buffer_unregister(off, &received_msg->payload.buffer_unreg, connection);
            break;
        case MSG_SHUTDOWN:
            DOCA_LOG_INFO("Received shutdown message from client");
            /* Don't stop the service - just acknowledge the client is disconnecting */
            /* The service should stay running and accept new connections */
            DOCA_LOG_INFO("Client disconnected, waiting for next connection...");
            break;
    }
    comch_utils_progress_connection(connection);
}

doca_error_t dpu_offloader_init(dpu_offloader_t **offloader, const char *server_addr) {
    dpu_offloader_t *off;
    struct comch_cfg *comch_cfg;
    struct doca_ctx *ctx;
    doca_error_t result;
    char pci_addr[PCI_ADDR_LEN] = "03:00.0"; /* Default fallback */
    char rep_pci_addr[PCI_ADDR_LEN] = {0};
    struct doca_devinfo_rep *rep_info;
    
    off = calloc(1, sizeof(dpu_offloader_t));
    if (server_addr) strcpy(off->server_name, server_addr);
    
    /* Always use auto-detection for the DMA device */
    result = open_dma_device(&off->dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_WARN("Auto-detect failed, attempting fallback to %s", pci_addr);
        result = open_doca_device_with_pci(pci_addr, NULL, &off->dev);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to open DOCA device: %s", doca_error_get_descr(result));
            free(off);
            return result;
        }
    }

    /* Log the device PCI address being used */
    {
        char dev_pci_addr[PCI_ADDR_LEN];
        struct doca_devinfo *devinfo = doca_dev_as_devinfo(off->dev);
        if (doca_devinfo_get_pci_addr_str(devinfo, dev_pci_addr) == DOCA_SUCCESS) {
            DOCA_LOG_INFO("Using DMA device at PCI address: %s", dev_pci_addr);
        }
    }
    
    /* Open representor if running as server */
    if (server_addr) {
        /* Use custom representor opener that scans widely */
        result = open_representor_custom(off->dev, NULL, &off->dev_rep);
        
        if (result == DOCA_SUCCESS && off->dev_rep) {
            rep_info = doca_dev_rep_as_devinfo(off->dev_rep);
            doca_devinfo_rep_get_pci_addr_str(rep_info, rep_pci_addr);
            DOCA_LOG_INFO("Using Representor PCI: %s", rep_pci_addr);
        } else {
            DOCA_LOG_WARN("No representor found! ComCh will not work properly.");
        }
    }
    
    /* Create and configure DMA engine */
    result = doca_dma_create(off->dev, &off->dma);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create DMA: %s", doca_error_get_descr(result));
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    ctx = doca_dma_as_ctx(off->dma);

    result = doca_pe_create(&off->pe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create PE: %s", doca_error_get_descr(result));
        doca_dma_destroy(off->dma);
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    result = doca_pe_connect_ctx(off->pe, ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to connect PE to ctx: %s", doca_error_get_descr(result));
        doca_pe_destroy(off->pe);
        doca_dma_destroy(off->dma);
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    result = doca_dma_task_memcpy_set_conf(off->dma, dma_completed_callback, dma_error_callback, MAX_QUEUE_DEPTH);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set DMA task conf: %s", doca_error_get_descr(result));
        doca_pe_destroy(off->pe);
        doca_dma_destroy(off->dma);
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    result = doca_ctx_start(ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start DMA context: %s", doca_error_get_descr(result));
        doca_pe_destroy(off->pe);
        doca_dma_destroy(off->dma);
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    result = doca_buf_inventory_create(MAX_BUFFERS * 2, &off->buf_inventory);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create buffer inventory: %s", doca_error_get_descr(result));
        doca_ctx_stop(ctx);
        doca_pe_destroy(off->pe);
        doca_dma_destroy(off->dma);
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    result = doca_buf_inventory_start(off->buf_inventory);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start buffer inventory: %s", doca_error_get_descr(result));
        doca_buf_inventory_destroy(off->buf_inventory);
        doca_ctx_stop(ctx);
        doca_pe_destroy(off->pe);
        doca_dma_destroy(off->dma);
        doca_dev_close(off->dev);
        free(off);
        return result;
    }

    /* Query maximum DMA buffer size */
    result = doca_dma_cap_task_memcpy_get_max_buf_size(doca_dev_as_devinfo(off->dev), &off->max_dma_buf_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get max DMA buffer size: %s", doca_error_get_descr(result));
        /* Use a conservative default */
        off->max_dma_buf_size = 2 * 1024 * 1024;  /* 2 MB default */
        DOCA_LOG_WARN("Using default max DMA buffer size: %lu bytes", off->max_dma_buf_size);
    } else {
        DOCA_LOG_INFO("Max DMA buffer size: %lu bytes (%.2f MB)",
                      off->max_dma_buf_size, off->max_dma_buf_size / (1024.0 * 1024.0));
    }

    DOCA_LOG_INFO("DMA engine and buffer inventory initialized successfully");

    if (server_addr) {
        /* If we have a representor, try to init ComCh */
        if (off->dev_rep) {
            const char *rep_pci_ptr = (rep_pci_addr[0] != '\0') ? rep_pci_addr : NULL;
            
            /* Important: ComCh utils needs the PCI address string of the *device* we are using */
            char dma_pci_addr[PCI_ADDR_LEN];
            struct doca_devinfo *dma_info = doca_dev_as_devinfo(off->dev);
            doca_devinfo_get_pci_addr_str(dma_info, dma_pci_addr);
            
            result = comch_utils_init(off->server_name, dma_pci_addr, rep_pci_ptr, off, NULL, server_recv_callback, &comch_cfg);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to init ComCh: %s", doca_error_get_descr(result));
                free(off);
                return result;
            }
            off->comch_cfg = comch_cfg;
            off->connection = comch_util_get_connection(comch_cfg);
        }
    }
    off->initialized = true;
    *offloader = off;
    return DOCA_SUCCESS;
}

doca_error_t dpu_offloader_init_standalone(dpu_offloader_t **offloader) {
    return dpu_offloader_init(offloader, NULL);
}

doca_error_t dpu_offloader_run(dpu_offloader_t *off) {
    if (!off || !off->initialized) return DOCA_ERROR_INVALID_VALUE;
    off->running = true;
    DOCA_LOG_INFO("DPU offloader service started");
    while (off->running && keep_running) {
        /* Always progress the DMA PE */
        doca_pe_progress(off->pe);
        
        /* Update connection pointer and progress comch - it may change when clients connect/disconnect */
        if (off->comch_cfg) {
            struct doca_comch_connection *current_conn = comch_util_get_connection(off->comch_cfg);
            
            /* Detect connection state changes */
            if (current_conn != off->connection) {
                if (current_conn == NULL && off->connection != NULL) {
                    DOCA_LOG_INFO("Client disconnected, waiting for new connection...");
                } else if (current_conn != NULL && off->connection == NULL) {
                    DOCA_LOG_INFO("New client connected");
                }
                off->connection = current_conn;
            }
            
            /* ALWAYS progress the comch PE so server can accept new clients */
            if (off->connection) {
                comch_utils_progress_connection(off->connection);
            } else {
                /* Even without a connection, progress the comch PE to accept new clients */
                comch_utils_progress(off->comch_cfg);
            }
        }
        
        usleep(100); 
    }
    return DOCA_SUCCESS;
}

doca_error_t dpu_offloader_get_stats(dpu_offloader_t *off, transfer_stats_t *stats) {
    if (!off || !stats) return DOCA_ERROR_INVALID_VALUE;
    memcpy(stats, &off->stats, sizeof(transfer_stats_t));
    return DOCA_SUCCESS;
}

void dpu_offloader_destroy(dpu_offloader_t *off) {
    if (!off) return;
    if (off->comch_cfg) comch_utils_destroy(off->comch_cfg);
    if (off->buf_inventory) doca_buf_inventory_destroy(off->buf_inventory);
    if (off->dma) doca_dma_destroy(off->dma);
    if (off->pe) doca_pe_destroy(off->pe);
    if (off->dev) doca_dev_close(off->dev);
    free(off);
    DOCA_LOG_INFO("DPU offloader destroyed");
}