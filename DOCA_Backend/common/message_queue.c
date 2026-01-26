#include "message_queue.h"
#include <string.h>

int message_queue_init(message_queue_t *queue) {
    if (!queue)
        return -1;
    
    memset(queue, 0, sizeof(message_queue_t));
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0)
        return -1;
    
    return 0;
}

int message_queue_push(message_queue_t *queue, const doca_message_t *msg) {
    if (!queue || !msg)
        return -1;
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->count >= MAX_QUEUE_SIZE) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    memcpy(&queue->messages[queue->tail], msg, sizeof(doca_message_t));
    queue->tail = (queue->tail + 1) % MAX_QUEUE_SIZE;
    queue->count++;
    
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

int message_queue_pop(message_queue_t *queue, doca_message_t *msg) {
    if (!queue || !msg)
        return -1;
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->count == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    memcpy(msg, &queue->messages[queue->head], sizeof(doca_message_t));
    queue->head = (queue->head + 1) % MAX_QUEUE_SIZE;
    queue->count--;
    
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

bool message_queue_is_empty(message_queue_t *queue) {
    if (!queue)
        return true;
    
    pthread_mutex_lock(&queue->mutex);
    bool empty = (queue->count == 0);
    pthread_mutex_unlock(&queue->mutex);
    
    return empty;
}

void message_queue_destroy(message_queue_t *queue) {
    if (!queue)
        return;
    
    pthread_mutex_destroy(&queue->mutex);
}
