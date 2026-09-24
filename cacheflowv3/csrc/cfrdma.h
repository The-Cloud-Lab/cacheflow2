/*
 * cfrdma: the CacheFlow v3 RDMA data plane.
 *
 * A small verbs + librdmacm library shared by both ends of the link:
 *   - the BlueField-3 Arm server, which owns the registered prefix-cache pool, and
 *   - the vLLM worker on the DGX Spark, which owns registered staging memory.
 *
 * One reliable-connected (RC) queue pair per endpoint. Either side can post
 * one-sided RDMA READ/WRITE batches against the peer's memory ("pull" mode is
 * client-initiated, "push" mode is server-initiated). A progress thread per
 * endpoint drains a software queue into the send queue and reaps completions, so
 * submissions of any size never block the caller.
 *
 * Completion tracking uses tickets: every work request gets a monotonically
 * increasing sequence number, and cf_post() returns the sequence number of the
 * last op in the batch. RC send queues complete in order, so a ticket is done
 * once the completed sequence reaches it.
 *
 * Every function is safe to call through Python ctypes (plain C types only).
 */
#ifndef CFRDMA_H
#define CFRDMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_OP_WRITE 0
#define CF_OP_READ 1

#define CF_MAX_PRIVATE_DATA 56

typedef struct cf_dev cf_dev;
typedef struct cf_mr cf_mr;
typedef struct cf_ep cf_ep;
typedef struct cf_listener cf_listener;

/* One RDMA READ/WRITE between local registered memory and peer memory. */
typedef struct cf_op {
    uint64_t laddr;
    uint64_t raddr;
    uint32_t lkey;
    uint32_t rkey;
    uint32_t len;
    uint32_t _pad;
} cf_op;

typedef struct cf_ep_stats {
    uint64_t ops_posted;
    uint64_t ops_completed;
    uint64_t bytes_posted;
    uint64_t sq_full_stalls;
} cf_ep_stats;

/* Last error message for this thread (never NULL). */
const char *cf_last_error(void);

/* Open the RDMA device that owns `ip` (a local address), or the device that
 * routes to `ip` when `is_peer` is non-zero. Returns NULL on failure. */
cf_dev *cf_dev_open(const char *ip, int is_peer);
const char *cf_dev_name(cf_dev *dev);
void cf_dev_close(cf_dev *dev);

/* Register [addr, addr+len) for local write and remote read/write. */
cf_mr *cf_reg_mr(cf_dev *dev, void *addr, size_t len);
uint32_t cf_mr_lkey(cf_mr *mr);
uint32_t cf_mr_rkey(cf_mr *mr);
void cf_dereg_mr(cf_mr *mr);

/* Page-aligned, pre-faulted anonymous memory, using 2MB hugepages if
 * `try_hugepages` is set and hugepages are available. *used_huge reports which. */
void *cf_alloc(size_t len, int try_hugepages, int *used_huge);
void cf_free(void *addr, size_t len);

/* Server side. cf_accept blocks up to timeout_ms (-1 = forever), returns NULL on
 * timeout (cf_last_error() == "timeout") or error. */
cf_listener *cf_listen(cf_dev *dev, const char *bind_ip, int port, int backlog);
cf_ep *cf_accept(cf_listener *l, int timeout_ms);
void cf_listener_close(cf_listener *l);

/* Client side. `pdata` (<= CF_MAX_PRIVATE_DATA bytes) is delivered to the server,
 * which reads it with cf_ep_private_data(). */
cf_ep *cf_connect(cf_dev *dev, const char *server_ip, int port, const void *pdata,
                  int plen, int timeout_ms);

/* Copy the connect-time private data into buf; returns its length. */
int cf_ep_private_data(cf_ep *ep, void *buf, int buflen);

/* Queue a batch of ops. *ticket receives the sequence number that completes
 * the batch. Returns 0 or a negative errno. */
int cf_post(cf_ep *ep, int opcode, const cf_op *ops, int n, uint64_t *ticket);

/* 0 = done, -ETIMEDOUT, or -EIO if the connection failed. timeout_ms < 0 = forever. */
int cf_wait(cf_ep *ep, uint64_t ticket, int timeout_ms);

/* 1 = done, 0 = pending, <0 = error. */
int cf_test(cf_ep *ep, uint64_t ticket);

/* 0 if healthy, otherwise the first failed ibv_wc_status (or -errno). */
int cf_ep_error(cf_ep *ep);
void cf_ep_get_stats(cf_ep *ep, cf_ep_stats *out);
void cf_ep_close(cf_ep *ep);

#ifdef __cplusplus
}
#endif
#endif /* CFRDMA_H */
