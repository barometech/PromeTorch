////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// nm_quad_load.h -                                                           //
//                                                                            //
//      Load and exchange library function declaration                        //
//      (for board NM_Quad)                                                   //
//                                                                            //
// Copyright (c) 2024 RC Module                                               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef __NM_QUAD_LOAD__H__
#define __NM_QUAD_LOAD__H__

#include <windows.h>

#define PLOAD_VERSION "7.2"
#define PLOAD_MAJOR_VERSION 7
#define PLOAD_MINOR_VERSION 2

#ifdef _MAKE_DLL
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

	// 32 bit unsigned NM memory element type
	typedef DWORD PL_Word;

	// 32 bit unsigned NM address type
	typedef DWORD PL_Addr;

	// Board descriptor
	typedef struct PL_Board PL_Board;

	// Core descriptor
	typedef struct PL_Access PL_Access;

	// Composite core number
	typedef struct PL_CoreNo
	{
		int nm_id;      // NM number (for any ARM: -1)
		int cluster_id; // Cluster number (for centeral ARM: -1)
	} PL_CoreNo;

	// Library functions return values (all library functions return result code)
	enum RetValue
	{
		PL_OK              = 0, // Ok
		PL_ERROR           = 1, // Error
		PL_TIMEOUT         = 2, // Timeout of wait for operation
		PL_FILE            = 3, // Failed find file for load
		PL_BADADDRESS      = 4, // Bad address ranges
		PL_AGAIN           = 5, // Try again
		PL_NOT_IMPLEMENTED = -1 // Not implemented
	};

	const PL_Word PROGRAM_NO        = 0x0;
	const PL_Word PROGRAM_PROGRESS  = 0x1;
	const PL_Word PROGRAM_FINISHED  = 0x2;
	const PL_Word PROGRAM_EXCEPTION = 0x4;

	//------------------//
	// Common functions //
	//------------------//

	// Get version of library
	DECLSPEC int PL_GetVersion(unsigned int *version_major, unsigned int *version_minor);

	// Set the channel as EDCL
	DECLSPEC int PL_SetChannelEDCL(const unsigned char host_mac_addr[6], const unsigned char board_mac_addr[6]);

	// Return number of detected boards in variable pointed by 'count'
	DECLSPEC int PL_GetBoardCount(unsigned int *count);

	// Create descriptor for board number 'index'
	// Return descriptor in variable pointed by 'board'
	// Boards index count from 0
	DECLSPEC int PL_GetBoardDesc(unsigned int index, PL_Board **board);

	// Close board descriptor and free descriptor memory
	DECLSPEC int PL_CloseBoardDesc(PL_Board *board);

	//-----------------//
	// Board functions //
	//-----------------//

	// Get serial number of board
	DECLSPEC int PL_GetSerialNumber(PL_Board *board, unsigned long long *serial_number);

	// Get version of firmware of board
	DECLSPEC int PL_GetFirmwareVersion(PL_Board *board, unsigned int *version_major, unsigned int *version_minor);

	// Send RESET signal to board and all processors on board
	DECLSPEC int PL_ResetBoard(PL_Board *board);

	// Load init code to board
	// Call this function after reset board and before loading user program to processors.
	DECLSPEC int PL_LoadInitCode(PL_Board *board);

	// Create descriptor for core
	// Return core descriptor in variable pointed by 'access'
	DECLSPEC int PL_GetAccess(PL_Board *board, PL_CoreNo *coreNo, PL_Access **access);

	// Close core descriptor
	DECLSPEC int PL_CloseAccess(PL_Access *access);

	//---------------------//
	// Processor functions //
	//---------------------//

	// Load user program on processor from memory and start execution
	// addrProgram - Address of program
	// sizeProgram - Size of program (in bytes)
	DECLSPEC int PL_LoadProgram(PL_Access *access, const void *addrProgram, unsigned int sizeProgram);

	// Load user program on processor from file and start execution
	DECLSPEC int PL_LoadProgramFile(PL_Access *access, const char *filename);

	// Load user program on processor from file (without start execution)
	DECLSPEC int PL_LoadProgramFileWithoutRun(PL_Access *access, const char *filename, PL_Addr *startAddr);

	// Start execution user program
	DECLSPEC int PL_RunProgram(PL_Access *access, PL_Addr startAddr);

	// Read array from shared memory
	// block   - Pointer to destination buffer in PC memory
	// address - Address of source array in board memory (for ARM - in bytes; for NM - in words)
	// len     - Size of array (in words)
	DECLSPEC int PL_ReadMemBlock(PL_Access *access, PL_Word *block, PL_Addr address, DWORD len);

	// Write array in shared memory
	// block   - Pointer to source array in PC memory
	// address - Address of destination array in board memory (for ARM - in bytes; for NM - in words)
	// len     - Size of array (in words)
	DECLSPEC int PL_WriteMemBlock(PL_Access *access, const PL_Word *block, PL_Addr address, DWORD len);

	// Read register (only for ARM core)
	// returnValue	- Value read from processor register
	// address	- Address of register to read in processor address space (in bytes)
	DECLSPEC int PL_ReadRegister(PL_Access *access, PL_Word *returnValue, PL_Addr address);

	// Write register (only for ARM core)
	// value	- Value to write in processor register
	// address	- Address of register to write in processor address space (in bytes)
	DECLSPEC int PL_WriteRegister(PL_Access *access, PL_Word value, PL_Addr address);

	// Barrier syncronization with program on board processor
	// value       - value sent to processor
	// returnValue - value received from processor
	DECLSPEC int PL_Sync(PL_Access *access, int value, int *returnValue);

	// Barrier syncronization with program on board processor
	// Additionally perform two values exchange
	// value       - value sent to processor
	// outAddress  - address sent to processor
	// outLen      - size sent to processor
	// returnValue - value received from processor
	// inAddress   - address received from processor
	// inLen       - size received from processor
	// Values return only if pointers are not NULL
	DECLSPEC int PL_SyncArray(
		PL_Access *access,  // Processor descriptor

		int value,          // Value sent to processor
		PL_Addr outAddress, // Address sent to processor
		PL_Word outLen,     // Size sent to processor

		int *returnValue,   // Value received from processor
		PL_Addr *inAddress, // Address received from processor
		PL_Word *inLen);    // Size received from processor

	// Set timeout in milliseconds for waiting function
	// Used in all PL_SyncXXX functions
	DECLSPEC int PL_SetTimeout(DWORD timeout);

	// Return status bits in variable 'status'
	DECLSPEC int PL_GetStatus(PL_Access *access, PL_Word *status);

	// Return user program return value in variable 'result'
	DECLSPEC int PL_GetResult(PL_Access *access, PL_Word *result);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // __NM_QUAD_LOAD__H__
