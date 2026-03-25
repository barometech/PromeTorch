#ifndef __IO_HOST_H__
#define __IO_HOST_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif
typedef struct PL_Access	PL_Access;
typedef struct IO_Service	IO_Service;

// call IO_Service constructor
IO_Service *IO_ServiceStart(
	const char* fName,
	PL_Access *access,
	FILE *file,
	void *ops,
	int *err);

	// call IO_Service constructor in binary mode
IO_Service * IO_ServiceStart_Binary(
	const char* fName,	// elf-code of core
	PL_Access* access,	// of the core
	FILE *_file,	// out print file 
	void * ops, // must be pointer on so library or NULL
	int * err );

// call IO_Service destructor
int IO_ServiceStop(
	IO_Service **service,
	int *err);

extern int print_mutex_lock(void);
extern int print_mutex_unlock(void);

// Set attributes of IO_Service destructor
//Attributes
enum IO_Service_Attributes {
	Att_Forse_Destruct = 1, // 0 - finish with error, if running target program
	                        // >0 - ignore running target program
	Att_Time_Destruct =  2, // in milliseconds (default: 10000)
	Att_Binary_Mode_Only = 3, //Has sense for Windows stdio. 
							// >0 - Files will opened in binary mode
							// 0 - Files will opened regularly
	Att_Poll_Interval = 4   // set poll inteval during dispatch loop in mseconds
};

int Set_IO_Serv_attribute(
	IO_Service *service,
	PL_Word number_attribute,
	PL_Word value_attribute
);
// return attribute value or -1 if failed arguments
int Get_IO_Serv_attribute(
	IO_Service *service,
	PL_Word number_attribute,
	PL_Word *value_attribute
);
// Read error result of ~IO_Service or BSP functions

//typedef struct { // error result
//	int function; // marker of BSP function 
//	int err = 0; // error value
//} io_service_fail_t;
// BSP function
enum BSP_function {
	IO_service_err = 0,
	IO_destruct_err = 1,
	BSP_read = 2,
	BSP_write = 3,
	BSP_status = 4,
	BSP_access = 5,
	BSP_RPC = 6,
	BSP_ERROR = 2000
};
// Error result  of funcions IO_ServiceStart and IO_ServiceStop
enum IO_SERV_FAIL {
 IO_SERV_PROG_RUN =1, // destructor is called when target program is still running
 IO_SERV_NOT_FINISHED =3, // io_service thread is not finished during set Time_Destruct
 IO_SERV_NO_RPC_SECTION = 4, // Target-prog has not rpc_services section
 IO_SERV_SECOND = 5, // Second IO_Service with the same core
 IO_SERV_NO_METHOD = 6, // No method to work with board
 IO_SERV_ERROR = 7, // Other errors in IO_Service
 IO_SERV_BSP = 100, // error occurred in BSP read/write functions

};
// return function_error of IO_Service running 
int Get_IO_Serv_Error(
	IO_Service *service,
	PL_Word *function,
	PL_Word *function_error
);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif // __IO_HOST_H__

