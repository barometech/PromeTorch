//------------------------------------------------------------------------
//
//  $Workfile:: mMatrixVecDev.h                       $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.2 $      $Date: 2005/06/23 15:15:15 $
//
//------------------------------------------------------------------------

#ifndef _MMATRIXVECTORDEV_H_INCLUDED_
#define _MMATRIXVECTORDEV_H_INCLUDED_


	/**
		\internal
		\addtogroup nmppmMul_mv_
	*/
	//! \{
void nmppmMul_mv_( nm64sc *pSrcMtr,  nm64sc *pSrcVec, nm64sc *pDstVec, int nHeight, int nWidth, void*tmp);
	//! \}

    /**
	\internal
    \defgroup MTR_ProdSelfV MTR_ProdSelfV
    \ingroup mMatrixVector
    \brief
        \ru Вычисляет произведение вектор-столбца на сопряженный к нему-же (получается квадратная матрица).
        \en ****************************************************************
		\~
    \param	pSrcVec		
        \ru Исходный вектор. 
        \en Source vec. 
		\~
    \param	nSize		
        \ru Размер вектора в элементах. 
        \en Vector size in elements. 
		\~
    \param	pTmp
        \ru Временный буфер размером nm64s[nSize*(nSize+1)*2]. 
        \en Temporary buffer with size nm64s[nSize*(nSize+1)*2]. 
		\~
    \retval	pDstMtr		
        \ru Результирующая матрица. 
        \en The result mtr. 
		\~
    \return \e void

	  \note
		  \ru Нет аналога.
		  \en ***********.
    
		\~
    \par
    \xmlonly
        <testperf>
          <param name="pSrcVec"> im0 im1 </param>
          <param name="pDstMtr"> im0 im1 </param>
          <param name="nSize"> 256 </param>
		  <size> nSize*nSize </size>
        </testperf>
        <testperf>
          <param name="pSrcVec"> im0 </param>
          <param name="pDstMtr"> im1 </param>
          <param name="nSize"> 8 128 256 </param>
		  <size> nSize*nSize </size>
        </testperf>
    \endxmlonly
    */
    //! \{
//void MTR_ProdSelfV( nm64sc *pSrcVec, nm64sc *pDstMtr, int nSize);
void MTR_ProdSelfV( nm64sc *pSrcVec, nm64sc *pDstMtr, int nSize, void* pTmp);
//void MTR_ProdSelfV( nm32sc *pSrcVec, nm64sc *pDstMtr, int nSize);//pc version is not available
    //! \}

    /**
	\internal
    \defgroup nmppmMul_mv__Zero nmppmMul_mv__Zero
    \ingroup mMatrixVector
    \brief
        \ru Вычисляет произведение матрицы на вектор-столбец, в котором nQuantity ненулевых элементов,
				    начиная с номера nStart.
        \en ******************************
		\~
    \param	pSrcMtr		
        \ru Исходная матрица. 
        \en Source mtr. 
		\~
    \param	pSrcVec		
        \ru Исходный вектор. 
        \en Source vec. 
		\~
    \param	nStart		
        \ru Номер первого ненулевого элемента. 
        \en *********************************. 
		\~
    \param	nQuantity		
        \ru Число ненулевых элементов. 
        \en *********************************. 
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
          <param name="pSrcVec"> im0 im1 </param>
          <param name="pSrcMtr"> im0 im1 </param>
          <param name="pDstVec"> im2 im3 </param>
          <param name="tmp"> im3 im1 </param>
          <param name="nStart"> 1 </param>
          <param name="nQuantity"> 12 </param>
          <param name="nHeight"> 128 </param>
          <param name="nWidth"> 256 </param>
					<size> nHeight*nWidth </size>
        </testperf>
        <testperf>
          <param name="pSrcVec"> im0 </param>
          <param name="pSrcMtr"> im1 </param>
          <param name="pDstVec"> im2 </param>
          <param name="tmp"> im3 </param>
          <param name="nStart"> 1 </param>
          <param name="nQuantity"> 12 </param>
          <param name="nHeight"> 128 </param>
          <param name="nWidth"> 16 128 256 </param>
					<size> nHeight*nWidth </size>
        </testperf>

    \endxmlonly
    */

    //! \{
void nmppmMul_mv__Zero( nm64sc *pSrcMtr,  nm64sc *pSrcVec, nm64sc *pDstVec, int nStart, int nQuantity, int nHeight, int nWidth, void *tmp);
    //! \}

    //--------------------------------------------------------------------
    // \defgroup nmppsSub8_Abs nmppsSub8_Abs
    // \ingroup mMatrixVector
    // \{
    //--------------------------------------------------------------------
	//void nmppsSub8_Abs(nm32s* pSrcVec, nm32s* SrcN8, nm32s* pDstVec, int nSize);
    // \todo void MTR_SubV8_Abs(nm32s* pSrcMtr,int nSrcStride,nm32s* pSrcVec8,nm32s* pDstMtr,int pDstStride,int nHeight);
    // \}

    //--------------------------------------------------------------------
    // \defgroup nmppsSub16_Abs nmppsSub16_Abs
    // \ingroup MatrixVector
    // \{
    //--------------------------------------------------------------------
//void nmppsSub16_Abs(nm8s* pSrcVec1, nm8s* pSrcVec2, nm8s* pDstVec, int nSize);
//void nmppsSub16_Abs(nm32s* pSrcVec1, nm32s* pSrcVec2, nm32s* pDstVec, int nSize);
//void nmppsSub16_Abs(nm32s* pSrcVec1, nm32s* pSrcVec2, nm32s* pDstVec, int nSize);
// \todo void MTR_SubV16_Abs(nm8s* pSrcMtr,int nSrcStride, nm8s* pSrcVec16, nm8s* pDstMtr,int nDstStride, int nHeight);
// \todo void MTR_SubV16_Abs(nm32s* pSrcMtr,int nSrcStride, nm32s* pSrcVec16, nm32s* pDstMtr,int nDstStride, int nHeight);

    // \}
#endif
