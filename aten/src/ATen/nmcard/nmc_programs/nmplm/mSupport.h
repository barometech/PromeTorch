//------------------------------------------------------------------------
//
//  $Workfile:: mSupport.h              $
//
//  <Название библиотеки>
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.2 $      $Date: 2005/06/23 15:15:15 $
//
//! \if file_doc
//!
//! \file   mtrMalloc.h
//! \author Дмитрий Рыжов
//! \brief  Распределение памяти для изображений.
//!
//! \endif
//!
//------------------------------------------------------------------------
#ifndef __MSUPPORT_H
#define __MSUPPORT_H

#include "nmpp.h"

//*****************************************************************************

	/*
    *
    * \if Russian
    *     \defgroup mSupport Функции поддержки
    * \endif
    * \if English
    *     \defgroup mSupport Support functions
    * \endif
    * \ingroup mtr
	*/

//*****************************************************************************

    /**
    \defgroup MTR_Malloc MTR_Malloc
    \ingroup mSupport
    \brief
        \ru Распределение памяти для матриц библиотеки.
        \en Memory allocation for library matrixis.

		\~
        \ru Начало и конец распределяемой памяти выравнивается на начало
        64-х разрядного слова.
        \en Begin and end of the allocated memory are being
        aligned to 64-bit word.

		\~
    \~
    \param nHeight
        \ru Число строк в матрице.
        \en Number of rows in mtr.
		\~
    \param nWidth
        \ru Число столбцов в матрице.
        \en Number of columns in mtr.
		\~
    \param hint
        \ru Номер банка памяти. Может принимать значения
                        MEM_LOCAL, MEM_GLOBAL.
        \en Number of memory bank. Admissible values for memory bank
                        are MEM_LOCAL, MEM_GLOBAL.

		\~
    \note \ru Память, распределенная с помощью функций MTR_Malloc должна
              освобождаться с помощью функции MTR_Free.
          \en Memory allocated by function MTR_Malloc should be
              deallocated by function MTR_Free.
			  \~
    */
    //! \{
	/*
__INLINE__ void MTR_Malloc(nm1** pptr,   int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>6) +1, hint);}
__INLINE__ void MTR_Malloc(nm2s** pptr,  int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>5) +1, hint);}
__INLINE__ void MTR_Malloc(nm4s** pptr,  int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>4) +1, hint);}
__INLINE__ void MTR_Malloc(nm8s** pptr,  int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>3) +1, hint);}
__INLINE__ void MTR_Malloc(nm16s** pptr, int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>2) +1, hint);}
__INLINE__ void MTR_Malloc(nm32s** pptr, int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>1) +1, hint);}
__INLINE__ void MTR_Malloc(nm64s** pptr, int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth   )   , hint);}

__INLINE__ void MTR_Malloc(nm2u** pptr,  int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>5) +1, hint);}
__INLINE__ void MTR_Malloc(nm4u** pptr,  int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>4) +1, hint);}
__INLINE__ void MTR_Malloc(nm8u** pptr,  int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>3) +1, hint);}
__INLINE__ void MTR_Malloc(nm16u** pptr, int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>2) +1, hint);}
__INLINE__ void MTR_Malloc(nm32u** pptr, int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth>>1) +1, hint);}
__INLINE__ void MTR_Malloc(nm64u** pptr, int nHeight, int nWidth, int hint = MEM_LOCAL) { nmppsMalloc_64s((nm64s**)pptr, (nHeight*nWidth   )   , hint);}
*/
    //! \}

//*****************************************************************************

    /**
    \defgroup MTR_Free MTR_Free
    \ingroup mSupport
    \brief
        \ru Освобождение памяти для матриц.
        \en Memory deallocation for matrixis.

		\~
		\~
    \note \ru Данная функция должна вызываться только для
              векторов, распределенных с помощью функций
              MTR_Malloc.
          \en This function should be called only for matrixis
              allocated by MTR_Malloc functions.
			  \~
    */
    //! \{

__INLINE__ void MTR_Free(void* ptr) {nmppsFree(ptr);}
    //! \}

//*****************************************************************************

    /**
    \defgroup MTR_Addr MTR_Addr
    \ingroup mSupport
    \brief
        \ru Возвращает адрес ячейки памяти, содержащей указанный элемент.
        \en Returns address of memory cell that contains
        the mtr element.

		\~
        \ru Реализация для процессора NeuroMatrix возвращает адрес, выровненный
            в памяти на 32 бита.
        \en Implementation for NeuroMatrix processor returns an addres aligned
            with 32 bits in the memory.

		\~
		\~
    \param pMtr
        \ru Входная матрица.
        \en Input mtr.
		\~
    \param nWidth
        \ru Ширина таблицы в элементах.
        \en Matrix width in elements.

		\~
    \param nY
        \ru Номер строки.
        \en Row index.

		\~
	\param nX
        \ru Номер колонки.
        \en Column index.

		\~
    \return
        \ru Адрес ячейки памяти.
        \en Address of memory cell.

		\~
    \note
        \ru Для ускорения работы на PC возможно использование макроса
            ADDR(ptr, index), который раскрывается на PC как
            (ptr+index), а на NM как вызов функции MTR_Addr.
        \en To increase perfomance on PC it's possible to use macros
            ADDR(ptr, index) that equivalent to (ptr+index) expression.

		\~
    \par
    \xmlonly
        <testperf>
             <param name="pMTR"> im0 </param>
             <param name="nWidth"> 4 </param>
             <param name="nY"> 2 </param>
             <param name="nX"> 2 </param>
             <size> 1 </size>
        </testperf>
    \endxmlonly
    */
    //! \{
__INLINE__ nm1*   MTR_Addr_1  (nm1*   pMTR, int nWidth, int nY, int nX)	{	return (nm1*)  ((int*)pMTR+((nWidth*nY+nX)>>5));}
__INLINE__ nm2s*  MTR_Addr_2s (nm2s*  pMTR, int nWidth, int nY, int nX)	{	return (nm2s*) ((int*)pMTR+((nWidth*nY+nX)>>4));}
__INLINE__ nm4s*  MTR_Addr_4s (nm4s*  pMTR, int nWidth, int nY, int nX)	{	return (nm4s*) ((int*)pMTR+((nWidth*nY+nX)>>3));}
__INLINE__ nm8s*  MTR_Addr_8s (nm8s*  pMTR, int nWidth, int nY, int nX)	{	return (nm8s*) ((int*)pMTR+((nWidth*nY+nX)>>2));}
__INLINE__ nm16s* MTR_Addr_16s(nm16s* pMTR, int nWidth, int nY, int nX)	{	return (nm16s*)((int*)pMTR+((nWidth*nY+nX)>>1));}
__INLINE__ nm32s* MTR_Addr_32s(nm32s* pMTR, int nWidth, int nY, int nX)	{	return (nm32s*)((int*)pMTR+(nWidth*nY+nX));}
__INLINE__ nm64s* MTR_Addr_64s(nm64s* pMTR, int nWidth, int nY, int nX) {	return (nm64s*)((int*)pMTR+((nWidth*nY+nX)<<1));}

__INLINE__ nm2u*  MTR_Addr_2u (nm2u*  pMTR, int nWidth, int nY, int nX)	{	return (nm2u*) ((int*)pMTR+((nWidth*nY+nX)>>4));}
__INLINE__ nm4u*  MTR_Addr_4u (nm4u*  pMTR, int nWidth, int nY, int nX)	{	return (nm4u*) ((int*)pMTR+((nWidth*nY+nX)>>3));}
__INLINE__ nm8u*  MTR_Addr_8u (nm8u*  pMTR, int nWidth, int nY, int nX)	{	return (nm8u*) ((int*)pMTR+((nWidth*nY+nX)>>2));}
__INLINE__ nm16u* MTR_Addr_16u(nm16u* pMTR, int nWidth, int nY, int nX)	{	return (nm16u*)((int*)pMTR+((nWidth*nY+nX)>>1));}
__INLINE__ nm32u* MTR_Addr_32u(nm32u* pMTR, int nWidth, int nY, int nX)	{	return (nm32u*)((int*)pMTR+(nWidth*nY+nX));}
__INLINE__ nm64u* MTR_Addr_64u(nm64u* pMTR, int nWidth, int nY, int nX) {	return (nm64u*)((int*)pMTR+((nWidth*nY+nX)<<1));}

    //! \}
//*****************************************************************************

    /**
    \defgroup MTR_SetVal MTR_SetVal
    \ingroup mSupport
    \brief
        \ru Зыписывате число в элемент марицы.
        \en Set mtr element.

		\~

	\f[
        pMtr[nY][nX] =  nVal
    \f]
    \param pMtr
        \ru Матрица.
        \en Matrix.
		\~
    \param nWidth
        \ru Ширина матрицы в элементах
        \en Matrix width in elements
		\~
	\param nY
		\ru Номер строки
		\en Row index
		\~
	\param nX
		\ru Номер столбца
		\en Column index
		\~
    \param nVal
        \ru Значение элемента
        \en Value of the element.
		\~
    \return \e void

    \par
    \xmlonly
        <testperf>
             <param name="pMtr"> im0 </param>
			 <param name="nWidth"> 18 </param>
			 <param name="nY"> 2  </param>
			 <param name="nX"> 2  </param>
             <param name="nVal"> 3 </param>
             <size> 1 </size>
        </testperf>
    \endxmlonly
    */
    //! \{
__INLINE__ void MTR_SetVal_1  (nm1*  pMtr,   int nWidth,  int nY, int nX, int1b nVal)	{ nmppsPut_1  (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_2s (nm2s* pMtr,   int nWidth,  int nY, int nX, int2b nVal)	{ nmppsPut_2s (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_4s (nm4s* pMtr,   int nWidth,  int nY, int nX, int4b nVal)	{ nmppsPut_4s (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_8s (nm8s* pMtr,   int nWidth,  int nY, int nX, int8b nVal)	{ nmppsPut_8s (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_16s(nm16s* pMtr,  int nWidth,  int nY, int nX, int16b nVal)	{ nmppsPut_16s(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_32s(nm32s* pMtr,  int nWidth,  int nY, int nX, int32b nVal)	{ nmppsPut_32s(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_64s(nm64s* pMtr,  int nWidth,  int nY, int nX, int64b nVal)	{ nmppsPut_64s(pMtr, nY*nWidth+nX,nVal);}

__INLINE__ void MTR_SetVal_2u (nm2u* pMtr,	 int nWidth,  int nY, int nX, uint2b nVal)	{ nmppsPut_2u (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_4u (nm4u* pMtr,   int nWidth,  int nY, int nX, uint4b nVal)	{ nmppsPut_4u (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_8u (nm8u* pMtr,   int nWidth,  int nY, int nX, uint8b nVal)	{ nmppsPut_8u (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_16u(nm16u* pMtr,  int nWidth,  int nY, int nX, uint16b nVal)	{ nmppsPut_16u(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_32u(nm32u* pMtr,  int nWidth,  int nY, int nX, uint32b nVal)	{ nmppsPut_32u(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_SetVal_64u(nm64u* pMtr,  int nWidth,  int nY, int nX, uint64b nVal)	{ nmppsPut_64u(pMtr, nY*nWidth+nX,nVal);}
    //! \}

//*****************************************************************************

	/**
    \defgroup MTR_GetVal MTR_GetVal
    \ingroup mSupport
    \brief
        \ru Считывает значение элемента марицы.
        \en Get mtr element.

		\~

	\f[
        nVal=pMtr[nY][nX]
    \f]
    \param pMtr
        \ru Матрица.
        \en Matrix.
		\~
    \param nWidth
        \ru Ширина матрицы в элементах
        \en Matrix width in elements
		\~
	\param nY
		\ru Номер строки
		\en Row index
		\~
	\param nX
		\ru Номер столбца
		\en Column index
		\~
    \retval nVal
        \ru Значение элемента
        \en Value of the element.
		\~
    \return \e void

    \par
    \xmlonly
        <testperf>
             <param name="pMtr"> im0 </param>
			 <param name="nWidth"> 18 </param>
			 <param name="nY"> 2  </param>
			 <param name="nX"> 2  </param>
             <param name="nVal"> 5 </param>
             <size> 1 </size>
        </testperf>
    \endxmlonly
    */
    //! \{
__INLINE__ void MTR_GetVal_1  (nm1*  pMtr,   int nWidth,  int nY, int nX, int1b* nVal)  { nmppsGetVal_1  (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_2s (nm2s* pMtr,   int nWidth,  int nY, int nX, int2b* nVal)  { nmppsGetVal_2s (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_4s (nm4s* pMtr,   int nWidth,  int nY, int nX, int4b* nVal)  { nmppsGetVal_4s (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_8s (nm8s* pMtr,   int nWidth,  int nY, int nX, int8b* nVal)  { nmppsGetVal_8s (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_16s(nm16s* pMtr,  int nWidth,  int nY, int nX, int16b* nVal) { nmppsGetVal_16s(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_32s(nm32s* pMtr,  int nWidth,  int nY, int nX, int32b* nVal) { nmppsGetVal_32s(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_64s(nm64s* pMtr,  int nWidth,  int nY, int nX, int64b* nVal) { nmppsGetVal_64s(pMtr, nY*nWidth+nX,nVal);}

__INLINE__ void MTR_GetVal_2u (nm2u* pMtr,	 int nWidth,  int nY, int nX, uint2b* nVal) { nmppsGetVal_2u (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_4u (nm4u* pMtr,   int nWidth,  int nY, int nX, uint4b* nVal) { nmppsGetVal_4u (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_8u (nm8u* pMtr,   int nWidth,  int nY, int nX, uint8b* nVal) { nmppsGetVal_8u (pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_16u(nm16u* pMtr,  int nWidth,  int nY, int nX, uint16b* nVal){ nmppsGetVal_16u(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_32u(nm32u* pMtr,  int nWidth,  int nY, int nX, uint32b* nVal){ nmppsGetVal_32u(pMtr, nY*nWidth+nX,nVal);}
__INLINE__ void MTR_GetVal_64u(nm64u* pMtr,  int nWidth,  int nY, int nX, uint64b* nVal){ nmppsGetVal_64u(pMtr, nY*nWidth+nX,nVal);}


    //! \}

//*****************************************************************************

    /**
    \internal
    \defgroup MTR_GetCol MTR_GetCol
    \ingroup mSupport
    \brief
        \ru Вынимает из матрицы один столбец.
        \en ********************************.
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
		\param nCol
		    \ru Номер вынимаемого столбца.
		    \en *************************.
		\~
    \param	nHeight
        \ru Число строк исходной матрицы.
        \en Row count in source mtr.
		\~
    \param	nWidth
        \ru Число столбцов исходной матрицы.
        \en Column count in source mtr.
		\~
    \retval	pDstVec
        \ru Результирующий вектор.
        \en The result vec.
		\~
    \return \e void

    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtr"> im0 im1 </param>
             <param name="pDstVec"> im0 im1   </param>
             <param name="nCol"> 4 </param>
             <param name="nHeight"> 64 </param>
             <param name="nWidth"> 128 </param>
             <size> nWidth*nHeight </size>
        </testperf>
        <testperf>
             <param name="pSrcMtr"> im0 </param>
             <param name="pDstVec"> im1 </param>
             <param name="nCol"> 4 </param>
             <param name="nHeight"> 64 </param>
             <param name="nWidth"> 8 32 128 </param>
             <size> nWidth*nHeight </size>
        </testperf>
    \endxmlonly
    */
    //! \{
void MTR_GetCol( nm64sc *pSrcMtr, nm64sc *pDstVec, int nCol, int nHeight, int nWidth);
//void MTR_GetCol( nm16sc *pSrcMtr, nm16sc *pDstVec, int nCol, int nHeight, int nWidth); //pc version is not available!
    //! \}

#endif
