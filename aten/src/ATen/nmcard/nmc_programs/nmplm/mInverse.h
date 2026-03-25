//------------------------------------------------------------------------
//
//  $Workfile:: mInverse.h  $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//
//! \if file_doc
//!
//! \file   mtrInverse.h
//! \author Сергей Мушкаев
//! \brief  Обращение матрицы.
//!
//! \endif
//!
//------------------------------------------------------------------------

#ifndef _MINVERSE_H_INCLUDED_
#define _MINVERSE_H_INCLUDED_

//*****************************************************************************
#ifdef __cplusplus
		extern "C" {
#endif

    /**
    \defgroup MTR_fpResolve_Gauss MTR_fpResolve_Gauss
    \ingroup mInverse
    \brief
        \ru Решение системы линейных уравнений методом Гаусса.
        \en Solve linear system by Gauss method .

		\~

	\f[
		A \cdot x = b
	\f]

    \param	pSrcMtrA
        \ru Квадтратная матрица уравнения A.
        \en Square system mtr A.
		\~
    \param	pSrcVecB
        \ru Свободный вектор.
        \en Free vec b.
		\~
    \param	nSize
        \ru Размер матрицы.
        \en Matrix size.
		\~
    \retval	pDstVecX
        \ru Вектор решения x.
        \en Solution vec.
		\~
    \return \e void

    \restr
        \ru Выбор ведущего элемента не производится -
              возможно деление на нуль.
        \en The function implements Gauss method without
              leading element selection.
		\~
    \note \ru Функция в данный момент не оптимизирована.
          \en The function is not optimized yet.
		\~
    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtrA"> im1 im2 </param>
             <param name="pSrcVecB"> im1 im2 </param>
             <param name="pDstVecX"> im3 </param>
             <param name="nSize"> 3 6 </param>
             <size> nSize*nSize </size>
        </testperf>
    \endxmlonly
    *///--------------------------------------------------------------------
    //! \{
void MTR_fpResolve_Gauss(double* pSrcMtrA, double* pSrcVecB, double* pDstVecX, int nSize);
    //! \}


//*****************************************************************************

    /**
    \defgroup MTR_fpResolve_PivotGauss MTR_fpResolve_PivotGauss
    \ingroup mInverse
    \brief
        \ru Решение системы линейных уравнений методом Гаусса с выбором
        ведущего элемента.
        \en Solve system of linear equations by Gauss method with leading
        element selection.

		\~

	\f[
    	A \cdot X = B
    \f]
    \param	pSrcMtrAB
        \ru Матрица уравнения A, дополненная
                            справа вектором B.
        \en Equation mtr A augmented with
                            vec B from the right.
		\~
    \param	nSize
        \ru Высота матрицы.
        \en Matrix height.
		\~
    \retval	pDstVecX
        \ru Вектор решения X.
        \en Solution vec X.
		\~
    \return \e void

    \par
    \xmlonly
        <testperf>
             <param name="pSrcMtrAB"> im1 im2 </param>
             <param name="pDstVecX"> im1 im2 </param>
             <param name="nSize"> 3 6 </param>
             <size> nSize*nSize </size>
        </testperf>
    \endxmlonly
    */
    //! \{
void MTR_fpResolve_PivotGauss(double* pSrcMtrAB, double* pDstVecX, int nSize);
    //! \}



	/** \defgroup nmppmLUDecomp nmppmLUDecomp
	    \ingroup mInverse
	 *  \brief LU-разложение матрицы
	 *
	 *  \param [in] A  Матрица A
	 *  \param [out] L Левая диагональная матрица L
	 *  \param [out] U верхняя диагональная матрица  U
	 *  \param [in]  N Размерность матрицы
	 *
	 */
//! \{
void nmppmLUDecomp_64f (
	const double  * A,
	double        * L,
	double        * U,
	int			    N
);
//! \}

 //SLAY
// L*y = b;  U*x = y

	/** \defgroup nmppmLUResolve nmppmLUResolve
	    \ingroup mInverse
	 *  \brief Решение системы линейных уравнений из LU-разложения
	 *  L*y = b;  U*x = y
	 *  \param [in] L Левая диагональная матрица L
	 *  \param [in] U верхняя диагональная матрица  U
	 *  \param [in] b вектор b
	 *  \param [out] x вектор x
	 *  \param [out] y вектор y
	 *  \param [in] N размер
	 *
	 *  \details More details
	 */
//! \{
void nmppmLUResolve_64f (
	const double        * L,
	const double        * U,
	const double        * b,
	double              * x,
	double              * y,
	int				    N
);
//! \}

#ifdef __cplusplus
		};
#endif


#endif // _MINVERSE_H_INCLUDED_
