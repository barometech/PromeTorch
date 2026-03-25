//------------------------------------------------------------------------
//
//  $Workfile:: mMatrixVec $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.1 $      $Date: 2005/01/12 14:03:24 $
//
//! \if file_doc
//!
//! \file   mtrArithm.h
//! \author Сергей Мушкаев
//! \brief  Базовы операции над элементами матрицы.
//!
//! \endif
//!
//------------------------------------------------------------------------
#ifndef _MMATRIXVECTOR_H_INCLUDED_
#define _MMATRIXVECTOR_H_INCLUDED_

#ifdef __cplusplus
		extern "C" {
#endif

//*****************************************************************************

    /**
    \defgroup nmppmMul_mm nmppmMul_mm
    \ingroup mMatrixVector
    \brief
        \ru Умножение матрицы на матрицу.
        \en Matrix by mtr multiplication.
		\~
    \param	pSrcMtr1
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	pSrcMtr2
        \ru Матрица-множитель.
        \en Multiplier mtr.
		\~
    \param	nHeight1
        \ru Число строк исходной матрицы.
        \en Row count in source mtr.
		\~
    \param	nWidth1
        \ru Число столбцов исходной матрицы.
        \en Column count in source mtr.
		\~
    \param	nWidth2
        \ru Число столбцов матрицы множителя.
        \en Column count in multiplier mtr.
		\~
    \param	nStride
        \ru Адресное смещение между строками матрицы.
        \en Address offset between rows.
		\~
    \param	bPlusDst
        \ru Флаг - указание прибавить к произведению прежнее значение Dst
        \en Flag - add old Dst to the product.
		\~
    \retval	pDstMtr
        \ru Результирующая матрица.
        \en The result mtr.
		\~
    \return \e void

    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtr1">im1 im2 im3</param>
             <param name="pSrcMtr2">im1 im2 im3</param>
             <param name="pDstMtr"> im1 im2 im3</param>
             <param name="nHeight1"> 8 16 64 </param>
             <param name="nWidth2"> 64 </param>
             <param name="nWidth1"> 32 64 </param>
             <size> nWidth1*nHeight1*nWidth2 </size>
        </testperf>
        <testperf>
             <param name="pSrcMtr1"> im1  </param>
             <param name="pSrcMtr2"> im2  </param>
             <param name="pDstMtr">  im3 </param>
             <param name="nWidth1">  32 64 </param>
             <param name="nHeight1"> 8 32 64 </param>
             <param name="nWidth2">  8 8 16</param>
             <size> nWidth1*nHeight1*nWidth2 </size>
        </testperf>
    \endxmlonly
    */
    //! \{
void nmppmMul_mm_2s64s ( nm2s* pSrcMtr1,  int nHeight1, int nWidth1,  nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);
void nmppmMul_mm_4s64s ( nm4s* pSrcMtr1,  int nHeight1, int nWidth1,  nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);
		
void nmppmMul_mm_8s8s  ( nm8s* pSrcMtr1,  int nHeight1, int nWidth1,  nm8s* pSrcMtr2,  nm8s*  pDstMtr, int nWidth2);
void nmppmMul_mm_8s16s ( nm8s* pSrcMtr1,  int nHeight1, int nWidth1,  nm16s* pSrcMtr2, nm16s* pDstMtr, int nWidth2);
void nmppmMul_mm_8s32s ( nm8s* pSrcMtr1,  int nHeight1, int nWidth1,  nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
void nmppmMul_mm_8s64s ( nm8s* pSrcMtr1,  int nHeight1, int nWidth1,  nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);

void nmppmMul_mm_16s16s( nm16s* pSrcMtr1, int nHeight1, int nWidth1,  nm16s* pSrcMtr2, nm16s* pDstMtr, int nWidth2);
void nmppmMul_mm_16s32s( nm16s* pSrcMtr1, int nHeight1, int nWidth1,  nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
void nmppmMul_mm_16s64s( nm16s* pSrcMtr1, int nHeight1, int nWidth1,  nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);

void nmppmMul_mm_32s32s( nm32s* pSrcMtr1, int nHeight1, int nWidth1,  nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
void nmppmMul_mm_32s64s( nm32s* pSrcMtr1, int nHeight1, int nWidth1,  nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);

void nmppmMul_mm_64s64s( nm64s* pSrcMtr1, int nHeight1, int nWidth1,  nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);


void nmppmMul_mm_32f  ( float* pSrcMtr1,  int nHeight1, int nStride1,
                        float* pSrcMtr2,  int nWidth1,  int nStride2,
                        float* pDstMtr,   int nWidth2,  int nStrideDst,  int bPlusDst );
void nmppmMul_mt_32f  ( float* pSrcMtr1,  int nHeight1, int nStride1,
                        float* pSrcMtr2,  int nWidth1,  int nStride2,
                        float* pDstMtr,   int nWidth2,  int nStrideDst,  int bPlusDst );

    //! \}

void nmppmMul_mm_colmajor_8s8s  (const nm8s* pSrcMtr1,  int nHeight1, int nWidth1, const nm8s* pSrcMtr2,  nm8s*  pDstMtr, int nWidth2);
void nmppmMul_mm_colmajor_8s16s (const nm8s* pSrcMtr1,  int nHeight1, int nWidth1, const nm16s* pSrcMtr2, nm16s* pDstMtr, int nWidth2);
void nmppmMul_mm_colmajor_8s32s (const nm8s* pSrcMtr1,  int nHeight1, int nWidth1, const nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
void nmppmMul_mm_colmajor_8s64s (const nm8s* pSrcMtr1,  int nHeight1, int nWidth1, const nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);

void nmppmMul_mm_colmajor_16s16s(const nm16s* pSrcMtr1, int nHeight1, int nWidth1, const nm16s* pSrcMtr2, nm16s* pDstMtr, int nWidth2);
void nmppmMul_mm_colmajor_16s32s(const nm16s* pSrcMtr1, int nHeight1, int nWidth1, const nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
void nmppmMul_mm_colmajor_16s64s(const nm16s* pSrcMtr1, int nHeight1, int nWidth1, const nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);

void nmppmMul_mm_colmajor_32s32s(const nm32s* pSrcMtr1, int nHeight1, int nWidth1, const nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
void nmppmMul_mm_colmajor_32s64s(const nm32s* pSrcMtr1, int nHeight1, int nWidth1, const nm64s* pSrcMtr2, nm64s* pDstMtr, int nWidth2);


	//*****************************************************************************

    /**
    \defgroup nmppmMul_mv_ nmppmMul_mv_
    \ingroup mMatrixVector
    \brief
        \ru Умножение матрицы на вектор.
        \en Matrix by vec multiplication.
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	pSrcVec
        \ru Вектор-множитель.
        \en Multiplier vec.
		\~
    \param	pSrcVec8
        \ru Вектор-множитель размерности 8.
        \en Multiplier vec of 8 elements.
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
          <param name="pSrcVec"> im0 im1 </param>
          <param name="pDstVec"> im2 im3 </param>
          <param name="nHeight"> 128 </param>
          <param name="nWidth"> 256 </param>
          <size> nHeight*nWidth </size>
        </testperf>
        <testperf>
          <param name="pSrcMtr"> im0 </param>
          <param name="pSrcVec"> im1 </param>
          <param name="pDstVec"> im2 </param>
          <param name="nHeight"> 128 </param>
          <param name="nWidth"> 16 128 256 </param>
          <size> nHeight*nWidth </size>
        </testperf>

    \endxmlonly
    */
    //! \{
void nmppmMul_mv_8s64s( nm8s* pSrcMtr,  nm64s* pSrcVec, nm64s* pDstVec, int nHeight, int nWidth);
void nmppmMul_mv_16s64s( nm16s* pSrcMtr,  nm64s* pSrcVec, nm64s* pDstVec, int nHeight, int nWidth);
void nmppmMul_mv_32s64s( nm32s* pSrcMtr,  nm64s* pSrcVec, nm64s* pDstVec, int nHeight, int nWidth);


    //! \}
void nmppmMul_mv_colmajor_8s64s (const nm8s*  pSrcMtr,const  nm64s* pSrcVec, nm64s* pDstVec, int nHeight, int nWidth);
void nmppmMul_mv_colmajor_16s64s(const nm16s* pSrcMtr,const  nm64s* pSrcVec, nm64s* pDstVec, int nHeight, int nWidth);
void nmppmMul_mv_colmajor_32s64s(const nm32s* pSrcMtr,const  nm64s* pSrcVec, nm64s* pDstVec, int nHeight, int nWidth);

	/**
    \defgroup nmppmMul_mv_8xH nmppmMul_mv_8xH
    \ingroup mMatrixVector
    \brief
        \ru Умножение матрицы на вектор.
        \en Matrix by vec multiplication.
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	pSrcVec
        \ru Вектор-множитель.
        \en Multiplier vec.
		\~
    \param	pSrcVec8
        \ru Вектор-множитель размерности 8.
        \en Multiplier vec of 8 elements.
		\~
    \param	nHeight
        \ru Число строк исходной матрицы.
        \en Row count in source mtr.
		\~
    \retval	pDstVec
        \ru Результирующий вектор.
        \en The result vec.
		\~
    \return \e void

    \par
    \xmlonly
        <testperf>
          <init> int nWidth=8; </init>
          <param name="pSrcMtr"> im0 im1 </param>
          <param name="pSrcVec"> im0 im1 </param>
          <param name="pDstVec"> im2 im3 </param>
          <param name="nHeight"> 128 </param>
          <size> nHeight*nWidth </size>
        </testperf>
        <testperf>
          <init> int nWidth=8; </init>
          <param name="pSrcMtr"> im0 </param>
          <param name="pSrcVec"> im1 </param>
          <param name="pDstVec"> im2 </param>
          <param name="nHeight"> 128 </param>
          <size> nHeight*nWidth </size>
        </testperf>

    \endxmlonly
    */
    //! \{
void nmppmMul_mv_8s16s_8xH( v8nm8s* pSrcMtr,  v8nm16s* pSrcVec, nm16s* pDstVec, int nHeight);
void nmppmMul_mv_16s16s_8xH( v8nm16s* pSrcMtr,  v8nm16s* pSrcVec, nm16s* pDstVec, int nHeight);
	//! \}
//*****************************************************************************

    /**
    \defgroup nmppmMul_mv__AddC nmppmMul_mv__AddC
    \ingroup mMatrixVector

 	\~
   \brief
        \ru Умножение матрицы на вектор с добавлением константы.
        \en Matrix by vec multiplication with addition of ant .
		\~
    \param	pSrcMtr
        \ru Исходная матрица.
        \en Source mtr.
		\~
    \param	pSrcVec
        \ru Вектор-множитель.
        \en Multiplier vec.
		\~
    \param	pSrcVec
        \ru Вектор-множитель размерности 2.
        \en Multiplier vec of 2 elements.
		\~
    \param	nAddVal
        \ru Константа.
        \en The ant.
		\~
    \param	nHeight
        \ru Число строк исходной матрицы. nHeight = [0, 2, 4, ...].
        \en Row count in source mtr. nHeight = [0, 2, 4, ...].
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
          <init> int nWidth=8; </init>
          <param name="pSrcMtr"> im0 im1 </param>
          <param name="pnSrcVec"> im0 im1  </param>
          <param name="pDstVec"> im2 im3 </param>
          <param name="nAddVal"> -1 </param>
          <param name="nHeight"> 128 </param>
          <size> nHeight </size>
        </testperf>
        <testperf>
          <init> int nWidth=8; </init>
          <param name="pSrcMtr">  im0 </param>
          <param name="pnSrcVec"> im1 </param>
          <param name="pDstVec">  im2 </param>
          <param name="nAddVal"> -1 </param>
          <param name="nHeight"> 32 128 1024 </param>
          <size> nHeight </size>
        </testperf>
    \endxmlonly
    */
    //! \{
void nmppmMul_mv__AddC(v2nm32s* pSrcMtr, v2nm32s* pnSrcVec, int nAddVal, nm32s* pDstVec, int nHeight);
    //! \}


//*****************************************************************************

    /**
    \defgroup MTR_ProdUnitV MTR_ProdUnitV
    \ingroup mMatrixVector
    \brief
        \ru Умножение матрицы на единичный вектор.
        \en Multiplication of a mtr by a unit vec.

		\~

	\f[
        pDstVec(i) = \sum_{j=0}^{w-1} pSrcMtr(i, j)
    \f]

    \ru Данная функция эквивалентна суммированию столбцов матрицы.
        Ширины матрицы, для которых имеется реализация данной функции
        указываются в ее названии.
    \en This function is equivalent to summation of mtr columns.
        Widths of a mtr for which this function is implemented
        are indicated in the mtr name.

		\~
		\~
    \param	pSrcMtr
        \ru Матрица.
        \en The mtr.
		\~
    \param	nHeight
        \ru Число строк матрицы. nHeight=[128,256,....]
        \en Row count in mtr. nHeight=[128,256,....]

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
        <param name="pDstVec"> im0 im1 </param>
        <param name="nHeight"> 128 </param>
      </testperf>

    \endxmlonly
    */
    //! \{
void MTR_ProdUnitV_16s_4xH ( v4nm16s* pSrcMtr, nm16s* pDstVec, int nHeight);
void MTR_ProdUnitV_16s_16xH( v16nm8s* pSrcMtr, nm16s* pDstVec, int nHeight);
    //! \}

//*****************************************************************************

    /**
	\internal
    \defgroup MTR_MulC_AddVsVc MTR_MulC_AddVsVc
    \ingroup mMatrixVector
    \brief
        \ru Умножение матрицы на константу с сумиированием вектора строки и вектора-столбца.
        \en Multiplication of a mtr by const with summation of vec-string and vec-column.

		\~

	\f[
        pDstVec(i) = \sum_{j=0}^{w-1} pSrcMtr(i, j)
    \f]


    \~

    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtr"> im0 im1 </param>
             <param name="pDstMtr"> im0 im1 </param>
             <param name="pSrcVecStr"> im2 im3 </param>
             <param name="pSrcVecCol"> im2 im3 </param>
             <param name="MulN"> 2 </param>
             <param name="nHeight"> 128 </param>
             <param name="nWidth"> 128 </param>
             <size> nHeight*nWidth </size>
        </testperf>
        <testperf>
             <param name="pSrcMtr"> im0 </param>
             <param name="pDstMtr"> im1 </param>
             <param name="pSrcVecStr"> im2 </param>
             <param name="pSrcVecCol"> im3 </param>
             <param name="MulN"> 2 </param>
             <param name="nHeight"> 128 </param>
             <param name="nWidth"> 8 32 128 </param>
             <size> nHeight*nWidth </size>
        </testperf>
    \endxmlonly
    *///--------------------------------------------------------------------
    //! \{
void MTR_MulC_AddVsVc(int MulN, nm32s* pSrcMtr, nm32s* pSrcVecStr, nm32s* pSrcVecCol, nm32s* pDstMtr, int nHeight, int nWidth);
    //! \}

#ifdef __cplusplus
		};
#endif

#endif // _MMATRIXVECTOR_H_INCLUDED_
