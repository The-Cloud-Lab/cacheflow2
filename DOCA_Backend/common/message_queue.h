#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include "protocol.h"
#include <pthread.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_QUEUE_SIZE 128

typedef struct {
    doca_message_t messages[MAX_QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
} message_queue_t;

/**
 * Initialize a message queue
 */
int message_queue_init(message_queue_t *queue);

/**
 * Push a message to the queue
 * Returns 0 on success, -1 if queue is full
 */
int message_queue_push(message_queue_t *queue, const doca_message_t *msg);

/**
 * Pop a message from the queue
 * Returns 0 on success, -1 if queue is empty
 */
int message_queue_pop(message_queue_t *queue, doca_message_t *msg);

/**
 * Check if queue is empty
 */
bool message_queue_is_empty(message_queue_t *queue);

/**
 * Destroy the message queue
 */
void message_queue_destroy(message_queue_t *queue);

#ifdef __cplusplus
}
#endif

#endif // MESSAGE_QUEUE_H
