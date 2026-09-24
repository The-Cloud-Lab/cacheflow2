/*
 * cfrdma: CacheFlow v3 RDMA data plane. See cfrdma.h for the model.
 */
#define _GNU_SOURCE
#include "cfrdma.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#define CF_HUGEPAGE (2UL << 20)
#ifdef MADV_POPULATE_WRITE
#define CF_MADV_POPULATE_WRITE MADV_POPULATE_WRITE
#else
#define CF_MADV_POPULATE_WRITE 23 /* Linux >= 5.14 */
#endif
#define CF_MAX_SQ_DEPTH 2048
#define CF_POST_CHAIN 32
#define CF_CQ_BATCH 32

static __thread char cf_errbuf[256] = "";

static void set_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cf_errbuf, sizeof(cf_errbuf), fmt, ap);
    va_end(ap);
}

const char *cf_last_error(void) { return cf_errbuf; }

/* ------------------------------------------------------------------------ */
/* Devices and memory                                                        */
/* ------------------------------------------------------------------------ */

struct cf_dev {
    struct rdma_event_channel *ch; /* keeps the librdmacm device referenced */
    struct rdma_cm_id *id;
    struct ibv_context *verbs;
    struct ibv_pd *pd;
    struct ibv_device_attr attr;
};

struct cf_mr {
    struct ibv_mr *mr;
};

static int parse_addr(const char *ip, int port, struct sockaddr_storage *out)
{
    struct addrinfo hints = {0}, *res = NULL;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV;
    int rc = getaddrinfo(ip, portstr, &hints, &res);
    if (rc) {
        set_err("bad address %s: %s", ip, gai_strerror(rc));
        return -1;
    }
    memcpy(out, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    return 0;
}

/* Wait for one CM event of the expected type on a channel. */
static int wait_cm_event(struct rdma_event_channel *ch, enum rdma_cm_event_type want,
                         int timeout_ms, struct rdma_cm_event **out)
{
    struct pollfd pfd = {.fd = ch->fd, .events = POLLIN};
    int rc = poll(&pfd, 1, timeout_ms);
    if (rc == 0) {
        set_err("timeout waiting for %s", rdma_event_str(want));
        return -ETIMEDOUT;
    }
    if (rc < 0) {
        set_err("poll: %s", strerror(errno));
        return -errno;
    }
    struct rdma_cm_event *ev;
    if (rdma_get_cm_event(ch, &ev)) {
        set_err("rdma_get_cm_event: %s", strerror(errno));
        return -errno;
    }
    if (ev->event != want) {
        set_err("expected %s, got %s (status %d)", rdma_event_str(want),
                rdma_event_str(ev->event), ev->status);
        rdma_ack_cm_event(ev);
        return -ECONNREFUSED;
    }
    if (out)
        *out = ev;
    else
        rdma_ack_cm_event(ev);
    return 0;
}

static void set_nonblock(int fd)
{
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
}

cf_dev *cf_dev_open(const char *ip, int is_peer)
{
    struct sockaddr_storage sa;
    if (parse_addr(ip, 0, &sa))
        return NULL;
    cf_dev *d = calloc(1, sizeof(*d));
    d->ch = rdma_create_event_channel();
    if (!d->ch || rdma_create_id(d->ch, &d->id, NULL, RDMA_PS_TCP)) {
        set_err("rdma_create_id: %s", strerror(errno));
        goto fail;
    }
    if (is_peer) {
        if (rdma_resolve_addr(d->id, NULL, (struct sockaddr *)&sa, 2000)) {
            set_err("rdma_resolve_addr(%s): %s", ip, strerror(errno));
            goto fail;
        }
        if (wait_cm_event(d->ch, RDMA_CM_EVENT_ADDR_RESOLVED, 3000, NULL))
            goto fail;
    } else if (rdma_bind_addr(d->id, (struct sockaddr *)&sa)) {
        set_err("rdma_bind_addr(%s): %s (is %s on an RDMA-capable netdev?)", ip,
                strerror(errno), ip);
        goto fail;
    }
    d->verbs = d->id->verbs;
    if (!d->verbs) {
        set_err("no RDMA device for %s", ip);
        goto fail;
    }
    if (ibv_query_device(d->verbs, &d->attr)) {
        set_err("ibv_query_device: %s", strerror(errno));
        goto fail;
    }
    d->pd = ibv_alloc_pd(d->verbs);
    if (!d->pd) {
        set_err("ibv_alloc_pd: %s", strerror(errno));
        goto fail;
    }
    return d;
fail:
    cf_dev_close(d);
    return NULL;
}

const char *cf_dev_name(cf_dev *d) { return ibv_get_device_name(d->verbs->device); }

void cf_dev_close(cf_dev *d)
{
    if (!d)
        return;
    if (d->pd)
        ibv_dealloc_pd(d->pd);
    if (d->id)
        rdma_destroy_id(d->id);
    if (d->ch)
        rdma_destroy_event_channel(d->ch);
    free(d);
}

cf_mr *cf_reg_mr(cf_dev *d, void *addr, size_t len)
{
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    struct ibv_mr *mr = ibv_reg_mr(d->pd, addr, len, access);
    if (!mr) {
        set_err("ibv_reg_mr(%p, %zu): %s", addr, len, strerror(errno));
        return NULL;
    }
    cf_mr *m = calloc(1, sizeof(*m));
    m->mr = mr;
    return m;
}

uint32_t cf_mr_lkey(cf_mr *m) { return m->mr->lkey; }
uint32_t cf_mr_rkey(cf_mr *m) { return m->mr->rkey; }

void cf_dereg_mr(cf_mr *m)
{
    if (!m)
        return;
    ibv_dereg_mr(m->mr);
    free(m);
}

void *cf_alloc(size_t len, int try_hugepages, int *used_huge)
{
    void *p = MAP_FAILED;
    if (used_huge)
        *used_huge = 0;
    if (try_hugepages) {
        size_t hlen = (len + CF_HUGEPAGE - 1) & ~(CF_HUGEPAGE - 1);
        p = mmap(NULL, hlen, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
        if (p != MAP_FAILED && used_huge)
            *used_huge = 1;
    }
    if (p == MAP_FAILED) {
        /* Transparent huge pages. The region must be 2MB aligned and marked
         * MADV_HUGEPAGE *before* it is faulted in; populating first (e.g.
         * MAP_POPULATE) leaves it on 4KB pages. 4KB pages cost the NIC one
         * IOMMU translation per page and capped inbound RDMA on the DGX Spark
         * at ~3.3 GB/s instead of ~11.5 GB/s. */
        size_t alen = (len + CF_HUGEPAGE - 1) & ~(CF_HUGEPAGE - 1);
        char *raw = mmap(NULL, alen + CF_HUGEPAGE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            set_err("mmap(%zu): %s", len, strerror(errno));
            return NULL;
        }
        char *a = (char *)(((uintptr_t)raw + CF_HUGEPAGE - 1) & ~(uintptr_t)(CF_HUGEPAGE - 1));
        if (a > raw)
            munmap(raw, a - raw);
        size_t tail = (size_t)((raw + alen + CF_HUGEPAGE) - (a + alen));
        if (tail)
            munmap(a + alen, tail);
        madvise(a, alen, MADV_HUGEPAGE);
        if (madvise(a, alen, CF_MADV_POPULATE_WRITE) != 0) {
            long pg = sysconf(_SC_PAGESIZE);
            for (size_t off = 0; off < alen; off += pg)
                a[off] = 0;
        }
        p = a;
    }
    return p;
}

void cf_free(void *addr, size_t len)
{
    /* Both allocation paths map a 2MB-rounded length. */
    if (addr)
        munmap(addr, (len + CF_HUGEPAGE - 1) & ~(CF_HUGEPAGE - 1));
}

/* ------------------------------------------------------------------------ */
/* Endpoints                                                                 */
/* ------------------------------------------------------------------------ */

typedef struct qop {
    cf_op op;
    uint64_t seq;
    uint8_t opcode;
    uint8_t batch_end;
} qop;

struct cf_ep {
    cf_dev *dev;
    struct rdma_event_channel *ch;
    struct rdma_cm_id *id;
    struct ibv_comp_channel *cc;
    struct ibv_cq *cq;
    int sq_depth;
    int wake_fd;

    pthread_t thr;
    int thr_started;
    pthread_mutex_t mu;
    pthread_cond_t done_cv;

    /* Software queue (ring buffer of ops not yet in the send queue). */
    qop *q;
    size_t q_cap, q_head, q_len;

    uint64_t next_seq;       /* sequence number the next op will get */
    uint64_t posted_seq;     /* last sequence number handed to the NIC */
    uint64_t completed_seq;  /* last sequence number known complete */
    uint64_t last_signaled;  /* last signaled sequence number posted */
    int error;
    int stop;

    uint8_t pdata[CF_MAX_PRIVATE_DATA];
    int plen;
    cf_ep_stats stats;
};

static int q_push(cf_ep *ep, const qop *o)
{
    if (ep->q_len == ep->q_cap) {
        size_t ncap = ep->q_cap ? ep->q_cap * 2 : 1024;
        qop *nq = malloc(ncap * sizeof(qop));
        if (!nq)
            return -ENOMEM;
        for (size_t i = 0; i < ep->q_len; i++)
            nq[i] = ep->q[(ep->q_head + i) % ep->q_cap];
        free(ep->q);
        ep->q = nq;
        ep->q_cap = ncap;
        ep->q_head = 0;
    }
    ep->q[(ep->q_head + ep->q_len) % ep->q_cap] = *o;
    ep->q_len++;
    return 0;
}

/* Move queued ops into the send queue while there is room. Caller holds mu. */
static void drain_locked(cf_ep *ep)
{
    struct ibv_send_wr wrs[CF_POST_CHAIN];
    struct ibv_sge sges[CF_POST_CHAIN];
    uint64_t signal_every = (uint64_t)ep->sq_depth / 2;

    while (ep->q_len && !ep->error) {
        int inflight = (int)(ep->posted_seq - ep->completed_seq);
        int room = ep->sq_depth - inflight;
        if (room <= 0) {
            ep->stats.sq_full_stalls++;
            return;
        }
        int n = 0;
        while (n < room && n < CF_POST_CHAIN && ep->q_len) {
            qop *o = &ep->q[ep->q_head];
            struct ibv_send_wr *wr = &wrs[n];
            memset(wr, 0, sizeof(*wr));
            sges[n].addr = o->op.laddr;
            sges[n].length = o->op.len;
            sges[n].lkey = o->op.lkey;
            wr->wr_id = o->seq;
            wr->sg_list = &sges[n];
            wr->num_sge = 1;
            wr->opcode = o->opcode == CF_OP_READ ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
            wr->wr.rdma.remote_addr = o->op.raddr;
            wr->wr.rdma.rkey = o->op.rkey;
            /* Signal batch ends (tickets) and often enough that a full send
             * queue always has a signaled op in flight to free slots. */
            if (o->batch_end || o->seq - ep->last_signaled >= signal_every) {
                wr->send_flags = IBV_SEND_SIGNALED;
                ep->last_signaled = o->seq;
            }
            wr->next = NULL;
            if (n)
                wrs[n - 1].next = wr;
            ep->stats.bytes_posted += o->op.len;
            ep->q_head = (ep->q_head + 1) % ep->q_cap;
            ep->q_len--;
            ep->posted_seq = o->seq;
            n++;
        }
        struct ibv_send_wr *bad = NULL;
        int rc = ibv_post_send(ep->id->qp, wrs, &bad);
        if (rc) {
            set_err("ibv_post_send: %s", strerror(rc));
            ep->error = -rc;
            pthread_cond_broadcast(&ep->done_cv);
            return;
        }
        ep->stats.ops_posted += n;
    }
}

static void wake(cf_ep *ep)
{
    uint64_t one = 1;
    ssize_t r = write(ep->wake_fd, &one, sizeof(one));
    (void)r;
}

/* Reap completions. Returns number reaped, or -1 on error. */
static int reap(cf_ep *ep)
{
    struct ibv_wc wc[CF_CQ_BATCH];
    int total = 0;
    for (;;) {
        int n = ibv_poll_cq(ep->cq, CF_CQ_BATCH, wc);
        if (n <= 0)
            return n < 0 ? -1 : total;
        pthread_mutex_lock(&ep->mu);
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                if (!ep->error)
                    ep->error = wc[i].status;
                continue;
            }
            if (wc[i].wr_id > ep->completed_seq) {
                ep->stats.ops_completed += wc[i].wr_id - ep->completed_seq;
                ep->completed_seq = wc[i].wr_id;
            }
        }
        drain_locked(ep);
        pthread_cond_broadcast(&ep->done_cv);
        pthread_mutex_unlock(&ep->mu);
        total += n;
    }
}

static void *progress_main(void *arg)
{
    cf_ep *ep = arg;
    struct pollfd pfd[3] = {
        {.fd = ep->cc->fd, .events = POLLIN},
        {.fd = ep->ch->fd, .events = POLLIN},
        {.fd = ep->wake_fd, .events = POLLIN},
    };
    while (1) {
        pthread_mutex_lock(&ep->mu);
        int stop = ep->stop;
        drain_locked(ep);
        pthread_mutex_unlock(&ep->mu);
        if (stop)
            break;

        if (reap(ep) > 0)
            continue;
        if (ibv_req_notify_cq(ep->cq, 0) == 0 && reap(ep) > 0)
            continue;

        if (poll(pfd, 3, 100) <= 0)
            continue;
        if (pfd[0].revents & POLLIN) {
            struct ibv_cq *cq;
            void *ctx;
            if (ibv_get_cq_event(ep->cc, &cq, &ctx) == 0)
                ibv_ack_cq_events(cq, 1);
        }
        if (pfd[1].revents & POLLIN) {
            struct rdma_cm_event *ev;
            if (rdma_get_cm_event(ep->ch, &ev) == 0) {
                if (ev->event == RDMA_CM_EVENT_DISCONNECTED ||
                    ev->event == RDMA_CM_EVENT_DEVICE_REMOVAL) {
                    pthread_mutex_lock(&ep->mu);
                    if (!ep->error)
                        ep->error = -ECONNRESET;
                    pthread_cond_broadcast(&ep->done_cv);
                    pthread_mutex_unlock(&ep->mu);
                }
                rdma_ack_cm_event(ev);
            }
        }
        if (pfd[2].revents & POLLIN) {
            uint64_t v;
            ssize_t r = read(ep->wake_fd, &v, sizeof(v));
            (void)r;
        }
    }
    return NULL;
}

static cf_ep *ep_new(cf_dev *dev)
{
    cf_ep *ep = calloc(1, sizeof(*ep));
    ep->dev = dev;
    pthread_mutex_init(&ep->mu, NULL);
    pthread_cond_init(&ep->done_cv, NULL);
    ep->wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    return ep;
}

/* Create CQ + QP on ep->id (which must already be on ep->dev's device). */
static int ep_setup_qp(cf_ep *ep)
{
    cf_dev *d = ep->dev;
    if (ep->id->verbs != d->verbs) {
        set_err("connection uses RDMA device %s but memory is registered on %s",
                ibv_get_device_name(ep->id->verbs->device), cf_dev_name(d));
        return -1;
    }
    ep->sq_depth = d->attr.max_qp_wr < CF_MAX_SQ_DEPTH ? d->attr.max_qp_wr : CF_MAX_SQ_DEPTH;
    ep->cc = ibv_create_comp_channel(d->verbs);
    if (!ep->cc) {
        set_err("ibv_create_comp_channel: %s", strerror(errno));
        return -1;
    }
    set_nonblock(ep->cc->fd);
    ep->cq = ibv_create_cq(d->verbs, ep->sq_depth + 16, ep, ep->cc, 0);
    if (!ep->cq) {
        set_err("ibv_create_cq: %s", strerror(errno));
        return -1;
    }
    struct ibv_qp_init_attr qa = {0};
    qa.send_cq = ep->cq;
    qa.recv_cq = ep->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = ep->sq_depth;
    qa.cap.max_recv_wr = 1;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.sq_sig_all = 0;
    if (rdma_create_qp(ep->id, d->pd, &qa)) {
        set_err("rdma_create_qp: %s", strerror(errno));
        return -1;
    }
    return 0;
}

static void conn_params(cf_dev *d, struct rdma_conn_param *cp, const void *pdata, int plen)
{
    memset(cp, 0, sizeof(*cp));
    cp->responder_resources = d->attr.max_qp_rd_atom;
    cp->initiator_depth = d->attr.max_qp_init_rd_atom;
    cp->retry_count = 7;
    cp->rnr_retry_count = 7;
    cp->private_data = pdata;
    cp->private_data_len = plen;
}

static int ep_start(cf_ep *ep)
{
    set_nonblock(ep->ch->fd);
    if (pthread_create(&ep->thr, NULL, progress_main, ep)) {
        set_err("pthread_create failed");
        return -1;
    }
    ep->thr_started = 1;
    return 0;
}

cf_ep *cf_connect(cf_dev *dev, const char *server_ip, int port, const void *pdata, int plen,
                  int timeout_ms)
{
    struct sockaddr_storage sa;
    if (plen < 0 || plen > CF_MAX_PRIVATE_DATA) {
        set_err("private data too long (%d > %d)", plen, CF_MAX_PRIVATE_DATA);
        return NULL;
    }
    if (parse_addr(server_ip, port, &sa))
        return NULL;
    cf_ep *ep = ep_new(dev);
    ep->ch = rdma_create_event_channel();
    if (!ep->ch || rdma_create_id(ep->ch, &ep->id, ep, RDMA_PS_TCP)) {
        set_err("rdma_create_id: %s", strerror(errno));
        goto fail;
    }
    if (rdma_resolve_addr(ep->id, NULL, (struct sockaddr *)&sa, timeout_ms)) {
        set_err("rdma_resolve_addr: %s", strerror(errno));
        goto fail;
    }
    if (wait_cm_event(ep->ch, RDMA_CM_EVENT_ADDR_RESOLVED, timeout_ms, NULL))
        goto fail;
    if (rdma_resolve_route(ep->id, timeout_ms)) {
        set_err("rdma_resolve_route: %s", strerror(errno));
        goto fail;
    }
    if (wait_cm_event(ep->ch, RDMA_CM_EVENT_ROUTE_RESOLVED, timeout_ms, NULL))
        goto fail;
    if (ep_setup_qp(ep))
        goto fail;
    struct rdma_conn_param cp;
    conn_params(dev, &cp, pdata, plen);
    if (rdma_connect(ep->id, &cp)) {
        set_err("rdma_connect: %s", strerror(errno));
        goto fail;
    }
    if (wait_cm_event(ep->ch, RDMA_CM_EVENT_ESTABLISHED, timeout_ms, NULL))
        goto fail;
    if (ep_start(ep))
        goto fail;
    return ep;
fail:
    cf_ep_close(ep);
    return NULL;
}

struct cf_listener {
    cf_dev *dev;
    struct rdma_event_channel *ch;
    struct rdma_cm_id *id;
};

cf_listener *cf_listen(cf_dev *dev, const char *bind_ip, int port, int backlog)
{
    struct sockaddr_storage sa;
    if (parse_addr(bind_ip, port, &sa))
        return NULL;
    cf_listener *l = calloc(1, sizeof(*l));
    l->dev = dev;
    l->ch = rdma_create_event_channel();
    if (!l->ch || rdma_create_id(l->ch, &l->id, l, RDMA_PS_TCP)) {
        set_err("rdma_create_id: %s", strerror(errno));
        goto fail;
    }
    if (rdma_bind_addr(l->id, (struct sockaddr *)&sa)) {
        set_err("rdma_bind_addr(%s:%d): %s", bind_ip, port, strerror(errno));
        goto fail;
    }
    if (rdma_listen(l->id, backlog)) {
        set_err("rdma_listen: %s", strerror(errno));
        goto fail;
    }
    return l;
fail:
    cf_listener_close(l);
    return NULL;
}

cf_ep *cf_accept(cf_listener *l, int timeout_ms)
{
    struct rdma_cm_event *ev = NULL;
    for (;;) {
        int rc = wait_cm_event(l->ch, RDMA_CM_EVENT_CONNECT_REQUEST, timeout_ms, &ev);
        if (rc == -ETIMEDOUT) {
            set_err("timeout");
            return NULL;
        }
        if (rc == 0)
            break;
        /* Some other event (e.g. a stale disconnect); keep waiting. */
    }
    struct rdma_cm_id *id = ev->id;
    cf_ep *ep = ep_new(l->dev);
    ep->id = id;
    id->context = ep;
    ep->plen = ev->param.conn.private_data_len < CF_MAX_PRIVATE_DATA
                   ? ev->param.conn.private_data_len
                   : CF_MAX_PRIVATE_DATA;
    if (ep->plen)
        memcpy(ep->pdata, ev->param.conn.private_data, ep->plen);
    rdma_ack_cm_event(ev);

    /* Give the connection its own event channel so its events don't mix
     * with other connection requests on the listener. */
    ep->ch = rdma_create_event_channel();
    if (!ep->ch || rdma_migrate_id(id, ep->ch)) {
        set_err("rdma_migrate_id: %s", strerror(errno));
        rdma_reject(id, NULL, 0);
        goto fail;
    }
    if (ep_setup_qp(ep)) {
        rdma_reject(id, NULL, 0);
        goto fail;
    }
    struct rdma_conn_param cp;
    conn_params(l->dev, &cp, NULL, 0);
    if (rdma_accept(id, &cp)) {
        set_err("rdma_accept: %s", strerror(errno));
        goto fail;
    }
    if (wait_cm_event(ep->ch, RDMA_CM_EVENT_ESTABLISHED, 5000, NULL))
        goto fail;
    if (ep_start(ep))
        goto fail;
    return ep;
fail:
    cf_ep_close(ep);
    return NULL;
}

void cf_listener_close(cf_listener *l)
{
    if (!l)
        return;
    if (l->id)
        rdma_destroy_id(l->id);
    if (l->ch)
        rdma_destroy_event_channel(l->ch);
    free(l);
}

int cf_ep_private_data(cf_ep *ep, void *buf, int buflen)
{
    int n = ep->plen < buflen ? ep->plen : buflen;
    memcpy(buf, ep->pdata, n);
    return n;
}

int cf_post(cf_ep *ep, int opcode, const cf_op *ops, int n, uint64_t *ticket)
{
    pthread_mutex_lock(&ep->mu);
    if (ep->error) {
        pthread_mutex_unlock(&ep->mu);
        set_err("endpoint failed (status %d)", ep->error);
        return -EIO;
    }
    for (int i = 0; i < n; i++) {
        if (ops[i].len == 0 || ops[i].len > (1U << 31)) {
            pthread_mutex_unlock(&ep->mu);
            set_err("op %d has invalid length %u", i, ops[i].len);
            return -EINVAL;
        }
    }
    for (int i = 0; i < n; i++) {
        qop o = {.op = ops[i], .seq = ++ep->next_seq, .opcode = (uint8_t)opcode,
                 .batch_end = (uint8_t)(i == n - 1)};
        if (q_push(ep, &o)) {
            pthread_mutex_unlock(&ep->mu);
            set_err("out of memory queueing ops");
            return -ENOMEM;
        }
    }
    if (ticket)
        *ticket = ep->next_seq;
    drain_locked(ep);
    int err = ep->error;
    pthread_mutex_unlock(&ep->mu);
    wake(ep);
    return err ? -EIO : 0;
}

int cf_wait(cf_ep *ep, uint64_t ticket, int timeout_ms)
{
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    if (timeout_ms >= 0) {
        dl.tv_sec += timeout_ms / 1000;
        dl.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
        if (dl.tv_nsec >= 1000000000L) {
            dl.tv_sec++;
            dl.tv_nsec -= 1000000000L;
        }
    }
    int rc = 0;
    pthread_mutex_lock(&ep->mu);
    while (ep->completed_seq < ticket && !ep->error) {
        if (timeout_ms < 0) {
            pthread_cond_wait(&ep->done_cv, &ep->mu);
        } else if (pthread_cond_timedwait(&ep->done_cv, &ep->mu, &dl) == ETIMEDOUT) {
            rc = -ETIMEDOUT;
            set_err("timeout waiting for ticket %lu (completed %lu)", (unsigned long)ticket,
                    (unsigned long)ep->completed_seq);
            break;
        }
    }
    if (!rc && ep->completed_seq < ticket) {
        rc = -EIO;
        set_err("endpoint failed (status %d)", ep->error);
    }
    pthread_mutex_unlock(&ep->mu);
    return rc;
}

int cf_test(cf_ep *ep, uint64_t ticket)
{
    pthread_mutex_lock(&ep->mu);
    int rc = ep->completed_seq >= ticket ? 1 : (ep->error ? -EIO : 0);
    pthread_mutex_unlock(&ep->mu);
    return rc;
}

int cf_ep_error(cf_ep *ep)
{
    pthread_mutex_lock(&ep->mu);
    int e = ep->error;
    pthread_mutex_unlock(&ep->mu);
    return e;
}

void cf_ep_get_stats(cf_ep *ep, cf_ep_stats *out)
{
    pthread_mutex_lock(&ep->mu);
    *out = ep->stats;
    pthread_mutex_unlock(&ep->mu);
}

void cf_ep_close(cf_ep *ep)
{
    if (!ep)
        return;
    if (ep->thr_started) {
        pthread_mutex_lock(&ep->mu);
        ep->stop = 1;
        pthread_mutex_unlock(&ep->mu);
        wake(ep);
        pthread_join(ep->thr, NULL);
    }
    if (ep->id) {
        if (ep->id->qp) {
            rdma_disconnect(ep->id);
            rdma_destroy_qp(ep->id);
        }
        rdma_destroy_id(ep->id);
    }
    if (ep->cq)
        ibv_destroy_cq(ep->cq);
    if (ep->cc)
        ibv_destroy_comp_channel(ep->cc);
    if (ep->ch)
        rdma_destroy_event_channel(ep->ch);
    if (ep->wake_fd >= 0)
        close(ep->wake_fd);
    pthread_mutex_lock(&ep->mu);
    pthread_cond_broadcast(&ep->done_cv);
    pthread_mutex_unlock(&ep->mu);
    free(ep->q);
    free(ep);
}
