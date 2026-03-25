//------------------------------------------------------------------------
//
//  $Workfile:: mInit.h  $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.1 $      $Date: 2005/01/12 14:03:24 $
//
//! \if file_doc
//!
//! \file   mtrInit.h
//! \author Сергей Мушкаев
//! \brief  Определение функций для инициализации и копирования.
//!
//! \endif
//!
//------------------------------------------------------------------------
#ifndef _MINIT_H_INCLUDED_
#define _MINIT_H_INCLUDED_


#ifdef __cplusplus
		extern "C" {
#endif



//*****************************************************************************

    //--------------------------------------------------------------------
    //! \ru
    //!     \defgroup mInit Инициализация и копирование
    //! \en
    //!     \defgroup mInit Initialization and copy
    //! \~
    //! \ingroup MTR_Group_
    //--------------------------------------------------------------------

//*****************************************************************************

    /**
    \defgroup nmppmCopyua nmppmCopyua
    \ingroup mInit
    \brief
        \ru Копирование подматрицы с невыровненной по границам 64- разрядного слова
			позиции в выровненную.
        \en Copying a submatrix from an unaligned to 64 bit word position to an alighed one.

		\~
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	nSrcStride
        \ru Ширина исходной матрицы в элементах.
        \en Source mtr width in elements.
		\~
    \param	nSrcOffset
        \ru Смещение в элементах от начала матрицы-источника.
        \en nShift in elements from the source mtr origin.
		\~
    \param	nDstStride
        \ru Ширина результирующей матрицы в элементах.
        \en Result mtr width in elements.
		\~
    \param	nHeight
        \ru Число строк подматрицы.
        \en Row count in submatrix.
		\~
    \param	nWidth
        \ru Число столбцов подматрицы.
        \en Column count in submatrix.

		\~
    \retval	pDstMtr
        \ru Результирующая матрица.
        \en The result mtr.

		\~
    \return \e void

    \restr
        \ru
			- Входная и выходная матрицы не должны перекрываться в памяти
            - Число столбцов копируемой подматрицы nWidth должна быть кратно числу
			  элементов в 64-х разрядном слове.
        \en
			- The input and output matrixes should not overlap in memory.
            - nWidth should be divisible by element count in 64-bit word.


		\~
    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtr"> im0 </param>
             <param name="nSrcStride"> 128 </param>
             <param name="nSrcOffset"> 0 </param>
             <param name="pDstMtr"> im1 </param>
             <param name="nDstStride"> 128 </param>
             <param name="nHeight"> 8 16 32 </param>
             <param name="nWidth"> 32 64 128 </param>
             <size>  nHeight * nWidth </size>
        </testperf>
    \endxmlonly
    */
    //! \{
void nmppmCopyua_8s (  nm8s* pSrcMtr, int nSrcStride, int nSrcOffset, nm8s* pDstMtr,  int nDstStride,int nHeight, int nWidth);
void nmppmCopyua_16s( nm16s* pSrcMtr, int nSrcStride, int nSrcOffset, nm16s* pDstMtr, int nDstStride,int nHeight, int nWidth);
void nmppmCopyua_32s( nm32s* pSrcMtr, int nSrcStride, int nSrcOffset, nm32s* pDstMtr, int nDstStride,int nHeight, int nWidth);
    //! \}

void nmppmCopy_32fc(
	double* SrcMtr,
	int     nSrcStride,
	double* DstMtr,
	int     nDstStride,
	int     nHeight,
	int     nWidth
);


//*****************************************************************************

    /**
    \defgroup MTR_Copyau MTR_Copyau
    \ingroup mInit
    \brief
        \ru Копирование подматрицы с выровненной по границе 64-х разрядных слов
			позиции в невыровненную позицию.
        \en Copying a submatrix from an aligned to 64 bit word position to an unalighed one.
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	nSrcStride
        \ru Ширина исходной матрицы в элементах.
        \en Source mtr width in elements.
		\~
    \retval	pDstMtr
        \ru Результирующая матрица.
        \en The result mtr.
		\~
    \param	nDstStride
        \ru Ширина результирующей матрицы в элементах.
        \en Result mtr width in elements.
		\~
    \param	nDstOffset
        \ru Индекс столбца (в элементах) куда производится вставка.
        \en nShift in elements from the source mtr origin.
		\~
    \param	nHeight
        \ru Число строк подматрицы.
        \en Row count in submatrix.
		\~
    \param	nWidth
        \ru Число столбцов подматрицы.
        \en Column count in submatrix.
		\~
    \return \e void

    \restr
        \ru
            - Входная и выходная матрицы не должны перекрываться в памяти
            - Число столбцов копируемой подматрицы nWidth должна быть кратной 64-битам.
        \en
            - The input and output matrixes should not overlap in memory.
            - The Column count of the copied submatrix nWidth should be divisible by 64 bits.

		\~
    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtr"> im0 </param>
             <param name="nSrcStride"> 128 </param>
             <param name="pDstMtr"> im1 </param>
             <param name="nDstStride"> 128 </param>
             <param name="nDstOffset"> 0 </param>
             <param name="nHeight"> 8 16 32 </param>
             <param name="nWidth"> 32 64 128 </param>
             <size>  nHeight* nWidth </size>
        </testperf>
    \endxmlonly

    */
    //! \{
void nmppmCopyau_8s (  nm8s* pSrcMtr, int nSrcStride, nm8s* pDstMtr,  int nDstStride, int nDstOffset,int nHeight, int nWidth);
void nmppmCopyau_16s( nm16s* pSrcMtr, int nSrcStride, nm16s* pDstMtr, int nDstStride, int nDstOffset,int nHeight, int nWidth);
void nmppmCopyau_32s( nm32s* pSrcMtr, int nSrcStride, nm32s* pDstMtr, int nDstStride, int nDstOffset,int nHeight, int nWidth);
    //! \}


//*****************************************************************************

    /**
    \defgroup MTR_Copy MTR_Copy
    \ingroup mInit
    \brief
        \ru Функции копирования прямоугольных областей памяти между двумерными массивами.
        \en Functions of copying rectangular memory areas between two-dimensional arrays.
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	nSrcStride
        \ru Ширина исходной матрицы в элементах.
        \en Source mtr width in elements.
		\~
    \param	nDstStride
        \ru Ширина результирующей матрицы в элементах.
        \en Result mtr width in elements.
		\~
    \param	nHeight
        \ru Число строк подматрицы.
        \en Row count in submatrix.
		\~
    \param	nWidth
        \ru Число столбцов подматрицы.
        \en Column count in submatrix.
		\~
    \retval	pDstMtr
        \ru Результирующая матрица.
        \en The result mtr.
		\~
    \return \e void

    \restr
        \ru Выходная матрица не может быть перезаписана (т.е. помещена непосредственно
            на место входной)
        \en The output mtr cannot be rewritten (i.e. placed directly to the
            input mtr location).

		\~
    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtr"> im0 </param>
             <param name="nSrcStride"> 64 </param>
             <param name="pDstMtr">  im1 </param>
             <param name="nDstStride"> 64 </param>
             <param name="nHeight"> 8 16 32 </param>
             <param name="nWidth"> 16 32 64 </param>
             <size> nHeight*nWidth </size>
        </testperf>
    \endxmlonly

    */
    //! \{
void nmppmCopy_1 (  nm1*  pSrcMtr, int nSrcStride,  nm1*  pDstMtr, int nDstStride, int nHeight, int nWidth);
void nmppmCopy_2s(  nm2s* pSrcMtr, int nSrcStride,  nm2s* pDstMtr, int nDstStride, int nHeight, int nWidth);
void nmppmCopy_4s(  nm4s* pSrcMtr, int nSrcStride,  nm4s* pDstMtr, int nDstStride, int nHeight, int nWidth);
void nmppmCopy_8s (  nm8s* pSrcMtr, int nSrcStride, nm8s* pDstMtr, int nDstStride, int nHeight, int nWidth);
void nmppmCopy_16s( nm16s* pSrcMtr, int nSrcStride, nm16s* pDstMtr, int nDstStride, int nHeight, int nWidth);
void nmppmCopy_32s( nm32s* pSrcMtr, int nSrcStride, nm32s* pDstMtr, int nDstStride, int nHeight, int nWidth);
void nmppmCopy_64s( nm64s* pSrcMtr, int nSrcStride, nm64s* pDstMtr, int nDstStride, int nHeight, int nWidth);
    //! \}


//*****************************************************************************


	/**
	\internal
	\defgroup MTR_Fill MTR_Fill
	\ingroup mInit
	\brief
		\ru Функция инициализации подматрицы постоянным значением.
		\en Function of submatrix initialization with a ant value.

		\~
	\~
	\param pMtr
        \ru Результирующая подматрица.
        \en The result submatrix.
		\~
	\param nVal
        \ru Константа.
        \en A ant.
		\~
	\param nMtrStride
		\ru Ширина всей матрицы \n
			(измеряется в 32-х разрядных словах).
         \en Width of whole mtr \n
            (measured in 32-bit words).
	\param nMtrHeight
        \ru Высота подматрицы.
        \en Submatrix height.
		\~
	\param nMtrWidth
		\ru Ширина подмартрицы в элементах
		\en Submatrix width in elements

		\~
    \par
    \xmlonly
        <testperf>
             <param name="pMtr"> im0 im1 </param>
             <param name="nVal"> 5 </param>
             <param name="nMtrStride"> 128 </param>
             <param name="nMtrHeight"> 8 16 32 64 </param>
             <param name="nMtrWidth"> 8 32 128 </param>
             <size>  nMtrHeight*nMtrWidth </size>
        </testperf>
    \endxmlonly

	*/
    //! \{
void MTR_Fill_8s(nm8s* pMtr,  int8b nVal, int nMtrStride, int nMtrHeight, int nMtrWidth);
    //! \}


#ifdef __cplusplus
		};
#endif


#endif // _MINIT_H_INCLUDED_
