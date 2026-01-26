#ifndef DOCA_KV_PROTOCOL_H
#define DOCA_KV_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Protocol version
#define PROTOCOL_VERSION 1

// Message types for communication between Host and DPU
typedef enum {
    MSG_INIT = 0,           // Initialize connection
    MSG_REGISTER_BUFFER,    // Register a new buffer for DMA
    MSG_TRANSFER_REQUEST,   // Request a DMA transfer (Host -> DPU, STORE)
    MSG_LOAD_REQUEST,       // Request a DMA load (DPU -> Host, LOAD)
    MSG_TRANSFER_COMPLETE,  // Transfer completion notification
    MSG_UNREGISTER_BUFFER,  // Unregister a buffer
    MSG_SHUTDOWN,           // Shutdown connection
    MSG_ACK,                // Acknowledgment
    MSG_ERROR               // Error notification
} message_type_t;

// Error codes
typedef enum {
    ERR_SUCCESS = 0,
    ERR_INVALID_BUFFER,
    ERR_DMA_FAILED,
    ERR_MEMORY_FULL,
    ERR_INVALID_MSG,
    ERR_TIMEOUT
} error_code_t;

// Buffer registration message
typedef struct {
    uint64_t buffer_id;      // Unique buffer identifier
    uint64_t host_addr;      // Host memory address
    size_t size;             // Buffer size in bytes
    uint8_t export_desc[1024]; // DOCA export descriptor (1024 bytes to handle larger descriptors)
    size_t export_desc_len;  // Length of export descriptor
} buffer_register_msg_t;

// Transfer request message (Host -> DPU, STORE operation)
typedef struct {
    uint64_t buffer_id;      // Buffer to transfer from
    uint64_t offset;         // Offset in buffer
    size_t length;           // Number of bytes to transfer
    uint64_t transfer_id;    // Unique transfer ID for tracking
} transfer_request_msg_t;

// Load request message (DPU -> Host, LOAD operation)
// Same structure as transfer_request but with reversed direction
typedef struct {
    uint64_t buffer_id;      // Buffer to load into (on host)
    uint64_t offset;         // Offset in buffer
    size_t length;           // Number of bytes to load
    uint64_t transfer_id;    // Unique transfer ID for tracking
} load_request_msg_t;

// Transfer complete message
typedef struct {
    uint64_t transfer_id;    // Transfer ID that completed
    error_code_t status;     // Completion status
    uint64_t bytes_transferred; // Actual bytes transferred
} transfer_complete_msg_t;

// Unregister buffer message
typedef struct {
    uint64_t buffer_id;      // Buffer to unregister
} buffer_unregister_msg_t;

// Generic message header
typedef struct {
    message_type_t type;     // Message type
    uint32_t version;        // Protocol version
    uint32_t sequence;       // Message sequence number
    size_t payload_size;     // Size of payload following header
} message_header_t;

// Generic message structure
typedef struct {
    message_header_t header;
    union {
        buffer_register_msg_t buffer_reg;
        transfer_request_msg_t transfer_req;
        load_request_msg_t load_req;
        transfer_complete_msg_t transfer_comp;
        buffer_unregister_msg_t buffer_unreg;
        error_code_t error;
    } payload;
} doca_message_t;

// Statistics structure
typedef struct {
    uint64_t total_transfers;
    uint64_t total_bytes;
    uint64_t failed_transfers;
    double avg_latency_us;
    double peak_bandwidth_gbps;
} transfer_stats_t;

#ifdef __cplusplus
}
#endif

#endif // DOCA_KV_PROTOCOL_H
