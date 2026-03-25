#ifndef __LLC_OPS_H__
#define __LLC_OPS_H__

#include <windows.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif
//typedef __u32			DWORD;
//typedef __u64			QWORD;

typedef DWORD			PL_Word;
typedef DWORD			PL_Addr;

typedef struct PL_Board	PL_Board;
typedef struct PL_Access	PL_Access;

typedef struct PL_CoreNo {
	int	nm_id;		// NM number (for any ARM: -1)
	int	cluster_id;	// Cluster number (for central ARM: -1)
} PL_CoreNo;

//typedef struct PL_CoreNo	PL_CoreNo;

enum RetValue {
	PL_OK			= 0, // OK
	PL_ERROR		= 1, // Error
	PL_TIMEOUT		= 2, // Time-out of wait for operation
	PL_FILE			= 3, // Cannot find file for load
	PL_BADADDRESS		= 4, // Bad address ranges
	PL_AGAIN		= 5, // Try again
	PL_NOT_IMPLEMENTED	= -1 // Not implemented
};

#define PROGRAM_NO		0x0
#define PROGRAM_PROGRESS	0x1
#define PROGRAM_FINISHED	0x2
#define PROGRAM_EXEPTION	0x4

typedef struct libload_ops  {
	int (*get_version)(unsigned int*, unsigned int*);
	int (*set_channel_EDCL)(const unsigned char[], const unsigned char[]); //for 08
	int (*get_board_count)(unsigned int*);
	int (*get_board_desc)(unsigned int, PL_Board**);
	int (*close_board_desc)(PL_Board*);
	int (*get_serial_number)(PL_Board*, unsigned long*);
	int (*get_firmware_version)(PL_Board*, unsigned int*, unsigned int*);
	int (*reset_board)(PL_Board*);
	int (*load_init_code)(PL_Board*);

	int (*close_access)(PL_Access*);
	int (*load_program)(PL_Access*, const void*, unsigned int);
	int (*load_program_file)(PL_Access*, const char*);
	int (*load_program_file_without_run)(PL_Access *,const char *,PL_Addr *);
	int(*run_program)(PL_Access *,PL_Addr );
	int (*read_mem_block)(PL_Access*, PL_Word*, PL_Addr, DWORD);
	int (*write_mem_block)(PL_Access*, const PL_Word*, PL_Addr, DWORD);
	int (*read_register)(PL_Access*, PL_Word*, PL_Addr); //for 08
	int (*write_register)(PL_Access*, PL_Word, PL_Addr);  //for 08
	int (*sync)(PL_Access*, int, int*);
	int (*sync_array)(PL_Access*, int, PL_Addr, PL_Word, int*, PL_Addr*, PL_Word*);
	int (*set_timeout)(DWORD);
	int (*get_status)(PL_Access*, PL_Word*);
	int (*get_result)(PL_Access*, PL_Word*);
// Special  part
	int (*get_access_07)(PL_Board*, unsigned int, PL_Access**);  //NM6407
	int (*get_access_08)(PL_Board*, PL_CoreNo*, PL_Access**); //NM6408
//	NULL
} libload_ops;

enum IO_Board_Type {
	IO_nm6407 = 0,
	IO_nm6408 = 1,
};

HMODULE LLC_ops_open(const char *boardName, libload_ops *wrap_ops, int *board_type);
int LLC_ops_close(HMODULE *handle );

#ifdef __cplusplus
} // extern "C"
#endif

#endif //__LLC_OPS_H__