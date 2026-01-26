/*
 * [UPDATED] DOCA_Backend/dpu/dpu_main.c
 * Added --pci argument support
 */
#include "dpu_offloader.h"
#include "../common/doca_common.h"
#include <stdio.h>
#include <string.h>
#include <getopt.h>

DOCA_LOG_REGISTER(DPU_MAIN);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -s, --server ADDR    Host server address (default: doca_kv_cache)\n");
    printf("  -S, --standalone     Run in standalone mode\n");
    printf("  -p, --pci ADDR       Force DPU PCI address (optional)\n");
    printf("  -h, --help           Show this help message\n");
}

int main(int argc, char **argv) {
    doca_error_t result;
    dpu_offloader_t *offloader = NULL;
    char server_addr[256] = "doca_kv_cache";
    bool standalone_mode = false;
    transfer_stats_t stats;
    
    static struct option long_options[] = {
        {"server", required_argument, 0, 's'},
        {"standalone", no_argument, 0, 'S'},
        {"pci", required_argument, 0, 'p'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "s:S:p:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 's':
                strncpy(server_addr, optarg, sizeof(server_addr) - 1);
                break;
            case 'S':
                standalone_mode = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
        }
    }
    
    result = doca_log_backend_create_standard();
    
    DOCA_LOG_INFO("========================================");
    DOCA_LOG_INFO("DOCA KV Cache Offloader - DPU Service");
    DOCA_LOG_INFO("========================================");
    
    if (standalone_mode) {
        DOCA_LOG_INFO("Mode: STANDALONE");
        result = dpu_offloader_init_standalone(&offloader);
    } else {
        DOCA_LOG_INFO("Mode: NORMAL");
        DOCA_LOG_INFO("Server address: %s", server_addr);
        result = dpu_offloader_init(&offloader, server_addr);
    }
    
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize: %s", doca_error_get_descr(result));
        return 1;
    }
    
    DOCA_LOG_INFO("Service initialized. Press Ctrl+C to stop.");
    result = dpu_offloader_run(offloader);
    
    dpu_offloader_get_stats(offloader, &stats);
    DOCA_LOG_INFO("Transfers: %lu, Failed: %lu", stats.total_transfers, stats.failed_transfers);
    
    dpu_offloader_destroy(offloader);
    return 0;
}