//------------------------------------------------------------------------
//
//  $Workfile:: vStat.h  $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.1 $      $Date: 2004/11/22 13:50:02 $
//
//! \if file_doc
//!
//! \file   vMinMax.h
//! \author Сергей Мушкаев
//! \brief  Определение функций поиска минимума и максимума
//!
//! \endif
//!
//------------------------------------------------------------------------

 
#ifndef _MSTAT_H_INCLUDED_
#define _MSTAT_H_INCLUDED_

#ifdef __cplusplus
		extern "C" {
#endif

    /**
	\internal
    \defgroup nmppmCrc_ nmppsCrc_
    \ingroup vStat
		\brief 
		    Подсчет циклического кода.
	  
		\param height
			Высота матрицы 

		\param width
			Ширина матрицы
		
		\param stride
			размер между строками в элементах
			
		\retval nCrc
			Циклический код.

		\~
    \par
    \xmlonly
		<testperf>
			<param name="pSrcVec"> im0 im1 		</param>
          <param name="crcAccumulator"> im2 		</param>
          <param name="nSize"> 2048 	</param>
				</testperf>
				<testperf>
          <param name="pSrcVec"> im0 		</param>
          <param name="crcAccumulator"> im2 		</param>
          <param name="nSize"> 8 128 1024 2048 	</param>
				</testperf>

    \endxmlonly
    */
    //! \{
			unsigned nmppmCrc_32u(const unsigned int* pSrcVec, int height, int width, int stride);
__INLINE__ 	unsigned nmppmCrc_32s(const nm32s       * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width   ,stride   ); }           
__INLINE__ 	unsigned nmppmCrc_64u(const nm64u       * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width<<1,stride<<1); }
__INLINE__ 	unsigned nmppmCrc_64s(const nm64s       * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width<<1,stride<<1); }
__INLINE__ 	unsigned nmppmCrc_16u(const nm16u       * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width>>1,stride>>1); }
__INLINE__ 	unsigned nmppmCrc_16s(const nm16s       * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width>>1,stride>>1); }
__INLINE__ 	unsigned nmppmCrc_8u (const nm8s        * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width>>2,stride>>2); }
__INLINE__ 	unsigned nmppmCrc_8s (const nm8s        * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width>>2,stride>>2); }
__INLINE__ 	unsigned nmppmCrc_4u (const nm4s        * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width>>3,stride>>3); }
__INLINE__ 	unsigned nmppmCrc_4s (const nm4s        * pSrcVec, int height, int width, int stride) {  return nmppmCrc_32u((const unsigned*)pSrcVec, height, width>>3,stride>>3); }
    //! \}
	
	unsigned nmppmCrcAcc_32u	(const unsigned int* pSrcVec, int height, int width, int stride, unsigned int* crc);
	
   
                                                                
//__INLINE__ unsigned  nmppsCrc_32f(nm32f* pSrcVec, int numBitsToClear, int nSize) {  return nmppsCrcMask_32u((unsigned*)pSrcVec,(-1<<numBitsToClear),nSize>>1); }
//																
//
//
//__INLINE__ unsigned nmppmCrcAcc_64s(nm64s* pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize<<1, crcAccumulator); }
//__INLINE__ unsigned nmppmCrcAcc_32s(nm32s* pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize,    crcAccumulator); }           
//__INLINE__ unsigned nmppmCrcAcc_16s(nm16s* pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize>>1, crcAccumulator); }
//__INLINE__ unsigned nmppmCrcAcc_8s (nm8s*  pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize>>2, crcAccumulator); }
//__INLINE__ unsigned nmppmCrcAcc_64u(nm64u* pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize<<1, crcAccumulator); }
//
//
//__INLINE__ unsigned nmppmCrcAcc_16u(nm16u* pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize>>1, crcAccumulator); }
//__INLINE__ unsigned nmppmCrcAcc_8u (nm8u*  pSrcVec, int nSize, unsigned int* crcAccumulator) {  return nmppsCrcAcc_32u((unsigned*)pSrcVec, nSize>>2, crcAccumulator); }
//
//unsigned nmppmCrcAcc_32f(const nm32f* pSrcVec, int numBitsToClear, int nSize,  unsigned int* crcAccumulator) ;                                                                                                          
//unsigned nmppmCrcAcc_64f(const nm64f* pSrcVec, int numBitsToClear, int nSize,  unsigned int* crcAccumulator) ;
//
//
	



	
#ifdef __cplusplus
		};
#endif

#endif // _MSTAT_H_INCLUDED_