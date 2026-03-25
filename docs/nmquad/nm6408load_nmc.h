#ifndef __NM6408LOAD_NMC_H__
#define __NM6408LOAD_NMC_H__

#ifdef __cplusplus
extern "C" {
#endif

struct ncl_ProcessorNo {
	int cluster_id;
	int core_id;
};

const char *ncl_getBoardName(void);
int ncl_getCoreID(void);
int ncl_getClusterID(void);
void ncl_getProcessorNo(struct ncl_ProcessorNo *ProcessorNo);

int ncl_hostSync(int value);
int ncl_hostSyncArray(int value, void *outArray, unsigned int outLen,
		      void **inArray, unsigned int *inLen);

void ncl_icache_dis(void);
void ncl_icache_ena(void);
void ncl_icache_inv(void);

void ncl_icache_gpa(void *addr);

#ifdef __cplusplus
};
#endif

#endif // __NM6408LOAD_NMC_H__
