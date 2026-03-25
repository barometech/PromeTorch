#ifndef __NM6408LOAD_ARM_H__
#define __NM6408LOAD_ARM_H__

#ifdef __cplusplus
extern "C" {
#endif

const char *acl_getBoardName(void);
int acl_getClusterID(void);

int acl_hostSync(int value);
int acl_hostSyncArray(int value, void *outArray, unsigned int outLen,
		      void **inArray, unsigned int *inLen);

void acl_defmmu_table_create(void);
int acl_mmu_setup(void *table);
void acl_mmu_free(void);

enum sync_act {
	SYNC_ACT_INVAL =	1, // invalidate
	SYNC_ACT_CLEAN =	2, // clean
	SYNC_ACT_FLUSH =	3, // clean & invalidate
};

void acl_flush_icache_all(void);
void acl_flush_icache_range(void *start_addr, void *end_addr);

void acl_sync_dcache_all(enum sync_act act);
void acl_sync_dcache_range(void *start_addr, void *end_addr, enum sync_act act);

void acl_coherent_range(void *start_addr, void *end_addr);

enum dma_dir {
	DMA_FROM_DEVICE =	1, // DMA write to memory
	DMA_TO_DEVICE =		2, // DMA read from memory
	DMA_BIDIRECTION =	3, // Unknown direction of access
};

int acl_sync_to_cpu(void *addr, unsigned int size, enum dma_dir dir);
int acl_sync_to_dma(void *addr, unsigned int size, enum dma_dir dir);

#ifdef __cplusplus
};
#endif

#endif // __NM6408LOAD_ARM_H__
