# Builds libcfrdma.so (the CacheFlow v3 RDMA data plane) into ../cacheflowv3/lib/.
# Needs libibverbs-dev and librdmacm-dev. Works unchanged on the DGX Spark and
# the BlueField-3 Arm (both aarch64), and on x86_64.
CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra -Wno-unused-parameter -fPIC -std=gnu11
OUT_DIR := ../cacheflowv3/lib
LIB := $(OUT_DIR)/libcfrdma.so

all: $(LIB)

$(LIB): cfrdma.c cfrdma.h
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) -shared -o $@ cfrdma.c -libverbs -lrdmacm -lpthread

clean:
	rm -f $(LIB)

.PHONY: all clean
