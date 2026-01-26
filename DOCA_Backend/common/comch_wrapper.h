#ifndef COMCH_WRAPPER_H
#define COMCH_WRAPPER_H

#include <doca_error.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct comch_cfg;

/**
 * Initialize a communication channel (client side - host)
 * 
 * @param server_name [in] - Server name to connect to
 * @param pci_addr [in] - PCI address of device
 * @param comch_cfg [out] - Communication channel configuration
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t comch_wrapper_client_init(const char *server_name,
                                        const char *pci_addr,
                                        struct comch_cfg **comch_cfg);

/**
 * Initialize a communication channel (server side - DPU)
 * 
 * @param server_name [in] - Server name to use
 * @param pci_addr [in] - PCI address of device
 * @param rep_pci_addr [in] - Representor PCI address
 * @param comch_cfg [out] - Communication channel configuration
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t comch_wrapper_server_init(const char *server_name,
                                        const char *pci_addr,
                                        const char *rep_pci_addr,
                                        struct comch_cfg **comch_cfg);

/**
 * Send a message over the channel
 * 
 * @param comch_cfg [in] - Communication channel configuration
 * @param msg [in] - Message to send
 * @param msg_len [in] - Length of message
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t comch_wrapper_send(struct comch_cfg *comch_cfg,
                                 const void *msg,
                                 size_t msg_len);

/**
 * Receive a message from the channel (blocking)
 * 
 * @param comch_cfg [in] - Communication channel configuration
 * @param msg [out] - Buffer to receive message into
 * @param msg_len [in/out] - Max length on input, actual length on output
 * @param timeout_ms [in] - Timeout in milliseconds (0 = no timeout)
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t comch_wrapper_recv(struct comch_cfg *comch_cfg,
                                 void *msg,
                                 size_t *msg_len,
                                 uint32_t timeout_ms);

/**
 * Destroy the communication channel
 * 
 * @param comch_cfg [in] - Communication channel configuration to destroy
 */
void comch_wrapper_destroy(struct comch_cfg *comch_cfg);

#ifdef __cplusplus
}
#endif

#endif // COMCH_WRAPPER_H
