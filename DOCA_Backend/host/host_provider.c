/*
 * [UPDATED] DOCA_Backend/host/host_provider.c
 * Ensure correct permissions for PCI export
 */
#include "host_provider.h"
#include "../common/doca_common.h"
#include "../common/comch_utils.h"
#include "../common/message_queue.h"
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

DOCA_LOG_REGISTER(HOST_PROVIDER);

/* Extended host provider with message queue */
typedef struct {
    host_provider_t base;
    struct comch_cfg *comch_cfg;
    message_queue_t msg_queue;
} host_provider_internal_t;

/* Helper function to find DOCA device by PCI address */
static doca_error_t open_doca_device_with_pci(const char *pci_addr, struct doca_dev **dev) {
    struct doca_devinfo **dev_list;
    uint32_t nb_devs;
    doca_error_t result;
    char devinfo_pci_addr[PCI_ADDR_LEN];
    
    result = doca_devinfo_create_list(&dev_list, &nb_devs);
    CHECK_DOCA_ERROR(result, "Failed to create device list");
    
    for (uint32_t i = 0; i < nb_devs; i++) {
        result = doca_devinfo_get_pci_addr_str(dev_list[i], devinfo_pci_addr);
        if (result != DOCA_SUCCESS) continue;
            
        if (strcmp(devinfo_pci_addr, pci_addr) == 0) {
            result = doca_dev_open(dev_list[i], dev);
            doca_devinfo_destroy_list(dev_list);
            return result;
        }
    }
    
    doca_devinfo_destroy_list(dev_list);
    DOCA_LOG_ERR("Device with PCI address %s not found", pci_addr);
    return DOCA_ERROR_NOT_FOUND;
}

/* Callback for receiving messages from DPU (client side) */
static void client_recv_callback(struct doca_comch_event_msg_recv *event,
                                  uint8_t *msg,
                                  uint32_t msg_len,
                                  struct doca_comch_connection *connection) {
    (void)event;
    
    host_provider_internal_t *prov_int = (host_provider_internal_t *)comch_utils_get_user_data(connection);
    if (!prov_int) return;
    
    doca_message_t received_msg;
    size_t copy_size = (msg_len < sizeof(doca_message_t)) ? msg_len : sizeof(doca_message_t);
    memcpy(&received_msg, msg, copy_size);
    
    message_queue_push(&prov_int->msg_queue, &received_msg);
}

doca_error_t host_provider_init(host_provider_t **provider, const char *pci_addr) {
    doca_error_t result;
    host_provider_internal_t *prov_int;
    host_provider_t *prov;
    
    prov_int = (host_provider_internal_t *)calloc(1, sizeof(host_provider_internal_t));
    prov = &prov_int->base;
    message_queue_init(&prov_int->msg_queue);
    
    strncpy(prov->pci_addr, pci_addr, PCI_ADDR_LEN - 1);
    snprintf(prov->server_name, sizeof(prov->server_name), "doca_kv_cache");
    
    DOCA_LOG_INFO("Opening DOCA device at PCI address: %s", pci_addr);
    result = open_doca_device_with_pci(pci_addr, &prov->dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open device at %s: %s", pci_addr, doca_error_get_descr(result));
        free(prov_int);
        return result;
    }
    DOCA_LOG_INFO("Successfully opened DOCA device");

    DOCA_LOG_INFO("Initializing ComCh connection to server '%s'", prov->server_name);
    result = comch_utils_init(prov->server_name, pci_addr, NULL, prov_int, client_recv_callback, NULL, &prov_int->comch_cfg);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to init ComCh: %s", doca_error_get_descr(result));
        doca_dev_close(prov->dev);
        free(prov_int);
        return result;
    }
    DOCA_LOG_INFO("ComCh connection established");
    
    prov->connection = comch_util_get_connection(prov_int->comch_cfg);
    prov->initialized = true;
    prov->connected = true;
    *provider = prov;
    return DOCA_SUCCESS;
}

doca_error_t host_provider_register_buffer(host_provider_t *provider, void *host_addr, size_t size, uint64_t *buffer_id) {
    doca_error_t result;
    host_buffer_t *buf;
    doca_message_t msg;
    host_provider_internal_t *prov_int = (host_provider_internal_t *)provider;
    
    buf = (host_buffer_t *)calloc(1, sizeof(host_buffer_t));
    buf->buffer_id = provider->num_buffers;
    buf->host_addr = host_addr;
    buf->size = size;
    
    result = doca_mmap_create(&buf->mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
        free(buf);
        return result;
    }

    result = doca_mmap_add_dev(buf->mmap, provider->dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add device to mmap: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_set_memrange(buf->mmap, host_addr, size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set memrange: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->mmap);
        free(buf);
        return result;
    }

    /* Set maximal permissions for DMA operations */
    result = doca_mmap_set_permissions(buf->mmap,
                              DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                              DOCA_ACCESS_FLAG_PCI_READ_WRITE |
                              DOCA_ACCESS_FLAG_RDMA_READ |
                              DOCA_ACCESS_FLAG_RDMA_WRITE);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set mmap permissions: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_start(buf->mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->mmap);
        free(buf);
        return result;
    }

    result = doca_mmap_export_pci(buf->mmap, provider->dev, (const void **)&buf->export_desc, &buf->export_desc_len);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Export PCI failed: %s", doca_error_get_descr(result));
        doca_mmap_destroy(buf->mmap);
        free(buf);
        return result;
    }

    DOCA_LOG_INFO("Exported mmap: addr=%p, size=%zu, export_desc_len=%zu",
                  host_addr, size, buf->export_desc_len);

    /* Check if export descriptor fits in our protocol message */
    if (buf->export_desc_len > sizeof(msg.payload.buffer_reg.export_desc)) {
        DOCA_LOG_ERR("Export descriptor too large! len=%zu, max=%zu",
                     buf->export_desc_len, sizeof(msg.payload.buffer_reg.export_desc));
        doca_mmap_destroy(buf->mmap);
        free(buf);
        return DOCA_ERROR_TOO_BIG;
    }
    
    memset(&msg, 0, sizeof(msg));
    msg.header.type = MSG_REGISTER_BUFFER;
    msg.header.sequence = provider->num_buffers;
    msg.header.payload_size = sizeof(buffer_register_msg_t);
    msg.payload.buffer_reg.buffer_id = buf->buffer_id;
    msg.payload.buffer_reg.host_addr = (uint64_t)host_addr;
    msg.payload.buffer_reg.size = size;
    msg.payload.buffer_reg.export_desc_len = buf->export_desc_len;
    memcpy(msg.payload.buffer_reg.export_desc, buf->export_desc, buf->export_desc_len);
    
    comch_utils_send(provider->connection, &msg, sizeof(message_header_t) + msg.header.payload_size);
    
    /* Wait for ACK */
    doca_message_t ack_msg;
    int retry = 0;
    while (retry++ < 1000) {
        comch_utils_progress_connection(provider->connection);
        if (message_queue_pop(&prov_int->msg_queue, &ack_msg) == 0 && ack_msg.header.type == MSG_ACK) {
            buf->registered = true;
            provider->buffers[provider->num_buffers++] = buf;
            *buffer_id = buf->buffer_id;
            return DOCA_SUCCESS;
        }
        usleep(1000);
    }
    return DOCA_ERROR_TIME_OUT;
}

doca_error_t host_provider_transfer(host_provider_t *provider, uint64_t buffer_id, uint64_t offset, size_t length, uint64_t *transfer_id) {
    doca_message_t msg;
    static uint64_t next_transfer_id = 1;

    *transfer_id = next_transfer_id++;

    memset(&msg, 0, sizeof(msg));
    msg.header.type = MSG_TRANSFER_REQUEST;
    msg.header.sequence = *transfer_id;
    msg.header.payload_size = sizeof(transfer_request_msg_t);
    msg.payload.transfer_req.buffer_id = buffer_id;
    msg.payload.transfer_req.offset = offset;
    msg.payload.transfer_req.length = length;
    msg.payload.transfer_req.transfer_id = *transfer_id;

    return comch_utils_send(provider->connection, &msg, sizeof(message_header_t) + msg.header.payload_size);
}

doca_error_t host_provider_load(host_provider_t *provider, uint64_t buffer_id, uint64_t offset, size_t length, uint64_t *transfer_id) {
    doca_message_t msg;
    static uint64_t next_load_id = 1000000;  /* Start from different range to distinguish from transfers */

    *transfer_id = next_load_id++;

    memset(&msg, 0, sizeof(msg));
    msg.header.type = MSG_LOAD_REQUEST;
    msg.header.sequence = *transfer_id;
    msg.header.payload_size = sizeof(load_request_msg_t);
    msg.payload.load_req.buffer_id = buffer_id;
    msg.payload.load_req.offset = offset;
    msg.payload.load_req.length = length;
    msg.payload.load_req.transfer_id = *transfer_id;

    DOCA_LOG_INFO("Sending LOAD request: buffer_id=%lu, offset=%lu, length=%zu, transfer_id=%lu",
                  buffer_id, offset, length, *transfer_id);

    return comch_utils_send(provider->connection, &msg, sizeof(message_header_t) + msg.header.payload_size);
}

doca_error_t host_provider_wait_transfer(host_provider_t *provider, uint64_t transfer_id, uint32_t timeout_ms) {
    host_provider_internal_t *prov_int = (host_provider_internal_t *)provider;
    doca_message_t comp_msg;
    int retries = timeout_ms * 10;
    
    while (retries-- > 0) {
        comch_utils_progress_connection(provider->connection);
        if (message_queue_pop(&prov_int->msg_queue, &comp_msg) == 0) {
            if (comp_msg.header.type == MSG_TRANSFER_COMPLETE && comp_msg.payload.transfer_comp.transfer_id == transfer_id) {
                return (comp_msg.payload.transfer_comp.status == ERR_SUCCESS) ? DOCA_SUCCESS : DOCA_ERROR_IO_FAILED;
            }
        }
        usleep(100);
    }
    return DOCA_ERROR_TIME_OUT;
}

doca_error_t host_provider_unregister_buffer(host_provider_t *provider, uint64_t buffer_id) {
    doca_error_t result;
    
    /* Find the buffer */
    for (int i = 0; i < provider->num_buffers; i++) {
        if (provider->buffers[i]->buffer_id == buffer_id) {
            host_buffer_t *buf = provider->buffers[i];
            
            /* Send unregister message to DPU */
            doca_message_t msg;
            memset(&msg, 0, sizeof(msg));
            msg.header.type = MSG_UNREGISTER_BUFFER;
            msg.header.sequence = buffer_id;
            msg.header.payload_size = sizeof(uint64_t);
            msg.payload.buffer_unreg.buffer_id = buffer_id;
            
            result = comch_utils_send(provider->connection, &msg, sizeof(message_header_t) + msg.header.payload_size);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_WARN("Failed to send unregister message: %s", doca_error_get_descr(result));
            }
            
            /* Destroy local mmap */
            if (buf->mmap) {
                doca_mmap_destroy(buf->mmap);
            }
            
            /* Remove from array by shifting remaining buffers */
            free(buf);
            for (int j = i; j < provider->num_buffers - 1; j++) {
                provider->buffers[j] = provider->buffers[j + 1];
            }
            provider->num_buffers--;
            
            DOCA_LOG_DBG("Unregistered buffer %lu", buffer_id);
            return DOCA_SUCCESS;
        }
    }
    
    DOCA_LOG_WARN("Buffer %lu not found", buffer_id);
    return DOCA_ERROR_NOT_FOUND;
}

doca_error_t host_provider_get_stats(host_provider_t *provider, transfer_stats_t *stats) {
    memcpy(stats, &provider->stats, sizeof(transfer_stats_t));
    return DOCA_SUCCESS;
}

void host_provider_destroy(host_provider_t *provider) {
    host_provider_internal_t *prov_int = (host_provider_internal_t *)provider;
    
    if (!provider) {
        return;
    }
    
    DOCA_LOG_DBG("Destroying provider, cleaning up %d buffers", provider->num_buffers);
    
    /* Clean up all registered buffers */
    while (provider->num_buffers > 0) {
        host_buffer_t *buf = provider->buffers[provider->num_buffers - 1];
        
        /* Destroy mmap */
        if (buf->mmap) {
            doca_mmap_destroy(buf->mmap);
            DOCA_LOG_DBG("Destroyed mmap for buffer %lu", buf->buffer_id);
        }
        
        free(buf);
        provider->num_buffers--;
    }
    
    /* Send disconnect notification but don't send shutdown (let DPU service keep running) */
    if (provider->connection) {
        /* Just progress a few times to flush any pending messages */
        for (int i = 0; i < 10; i++) {
            comch_utils_progress_connection(provider->connection);
            usleep(1000);  /* 1ms */
        }
        
        DOCA_LOG_DBG("Connection closing gracefully");
    }
    
    /* Destroy comm channel */
    if (prov_int->comch_cfg) {
        DOCA_LOG_DBG("Destroying comm channel");
        comch_utils_destroy(prov_int->comch_cfg);
    }
    
    /* Close device */
    if (provider->dev) {
        DOCA_LOG_DBG("Closing DOCA device");
        doca_dev_close(provider->dev);
    }
    
    DOCA_LOG_DBG("Provider destroyed successfully");
    free(provider);
}