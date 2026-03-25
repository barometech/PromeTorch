 /**
 *  \brief     nmblas (NeuroMatrix Basic Linear Algebra Subroutines)
 *  \details   This class is used to demonstrate a number of section commands.
 *  \author    Alexandr Bolotnikov 
 *  \version   1.0
 *  \date      2017-05-23T15:13:13
 *  \warning   
 *  \copyright (c) RC Module Inc.
 *  \file nmblas.h
 */
 
 
 //! \defgroup LEVEL1 BLASS-LEVEL1
 //! \ingroup NMBLAS
 //! \{
 
 //! \}

 //! \defgroup LEVEL2 BLASS-LEVEL2
 //! \ingroup NMBLAS
 //! \{
 
 //! \}


#ifndef _NMBLAS_H_INCLUDED_
#define _NMBLAS_H_INCLUDED_

#ifdef __cplusplus
		extern "C" {
#endif	


	/**
	 *  
	 *  \brief takes the sum of the absolute values
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX  storage spacing between elements of X
	    \par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
        \endxmlonly
	 *  \return the sum of the absolute values.
	 */
//! \{
double nmblas_dasum(
	const int N,
	const double *X,
	const int INCX
);
//! \}

enum nm_trans{nm_n=0,nm_t=1};


	/**
	 *  
	 *  \brief constant times a vector plus a vector
	 *  \ingroup LEVEL1	 	 
	 *  
 	 *  \param [in] N number of elements in input vector
	 *  \param [in] A specifies the scalar alpha
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	    \par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" A "> 5 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
void nmblas_daxpy(
	const int N,
	const double A,
	const double *X,
	const int INCX,
	double *Y,
	const int INCY
); 
//! \}

	 
	/**
	 *  \brief copies a vector, X, to a vector, Y
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly

	 */
//! \{
void nmblas_dcopy(
	const int N,
	const double* X,
	const int INCX,
	double* Y,
	const int INCY
);
//! \}

	/**
	 *  \brief  forms the dot product of two vectors.
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \return Return dot product of two vectors
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly

	 */
//! \{
double nmblas_ddot(
	const int N,
	const double* X,
	const int INCX,
	const double* Y,
	const int INCY
);
//! \}

	/**
	 *  \brief returns the euclidean norm of a vector via the function name, so that  SCNRM2 := sqrt( x**H*x )
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \return euclidean norm
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" x "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
double nmblas_dnrm2(
	const int N,
	const double *x,
	const int INCX
);
//! \}

	/**
	 *  \brief applies a plane rotation
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in,out] X  array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \param [in] C 
	 *  \param [in] S 
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <param name=" C "> 1 </param>
				 <param name=" S "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
void nmblas_drot(
	const int N,
	double *X,
	const int INCX,
	double *Y,
	const int INCY,
	const double C,
	const double S
); 
//! \}

	/**
	 *  \brief construct givens plane rotation
	 *  \ingroup LEVEL1	 	 
	 *  
	 *  \param [in] A 
	 *  \param [in] B 
	 *  \param [in] C 
	 *  \param [out] S 
	 */
//! \{
void nmblas_drotg(
	double *A, 
	double *B, 
	double *C, 
	double *S
);
//! \}

	/**
	 *  \brief  Apply a Given's rotation constructed by DROTMG.
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in,out] X  array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \param [in] param array, dimension (5). The  rotation  values  constructed  by SROTMG
	 *  
	 *  \details 
	 */
//! \{
void nmblas_drotm(
	const int N,
	double *X,
	const int INCX,
	double *Y,
	const int INCY,
	double *param
);
//! \} 

	/**
	 *  \brief scales a vector by a constant
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] A specifies the scalar alpha.
	 *  \param [in,out] X array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
				 <param name=" ALPHA "> 5 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
void nmblas_dscal(
	const int N,
	const double ALPHA,
	double*X,
	const int INCX
); 
//! \}

	/**
	 *  \brief Compute the inner product of two vectors with double precision accumulation.
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] B single precision scalar to be added to inner product
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX ) ) single precision vector with N elementsr
     *  \param [in] INCX storage spacing between elements of X
	 *  \param [in] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \return  Returns D.P. dot product accumulated in D.P., for S.P. SX and SY
     * DSDOT = sum for I = 0 to N-1 of  SX(LX+I*INCX) * SY(LY+I*INCY),
     * where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
     * defined in a similar way using INCY.
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly 
	 */
//! \{
double nmblas_dsdot(
	const int N,
	const float* X,
	const int INCX,
	const float* Y,
	const int INCY
);
//! \}

	/**
	 *  \brief interchanges two vectors
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in,out] X  array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \return 
	 *  
	 *  \details 
	 */
//! \{
void nmblas_dswap(
	const int N,
	double *X,
	const int INCX,
	double *Y,
	const int INCY
);
//! \}

	/**
	 *  \brief  returns the euclidean norm of a vector via the function name, so that   DZNRM2 := sqrt( x**H*x )
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X COMPLEX*16 array of DIMENSION at least ( 1 + ( m - 1 )*abs( INCX  )  )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \return  euclidean norm 
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
double nmblas_dznrm2(
	const int N,
	const double *X,
	const int INCX
);  
//! \}

	/**
	 *  \brief finds the index of the first element having maximum absolute value.
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX ))
	 *  \param [in] INCX storage spacing between elements of X
	 *  \return index of the first element having maximum absolute value.
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
int nmblas_idamax(
	const int N,
	const double* X,
	const int INCX
);
//! \}
//////////////////////////////////////////////////////level 1 single precition

	/**
	 *  \brief finds the index of the first element having maximum absolute value.
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \return index of the first element having maximum absolute value.
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 4096 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
int nmblas_isamax(
	const int N,
	const float* X,
	const int INCX
);
//! \}

	/**
	 *  \brief takes the sum of the absolute values
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] A specifies the scalar alpha
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \return sum of the absolute values
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
float nmblas_sasum(
	const int N,
	const float *X,
	const int INCX
);
//! \}
	
	/**
	 *  \brief  constant times a vector plus a vector
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] A specifies the scalar alpha
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	    \par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" A "> 5 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
void nmblas_saxpy(
    int N,
	const float A,
	const float *X,
	const int INCX,
	float *Y,
	const int INCY
);
//! \}

	/**
	 *  \brief returns the euclidean norm of a vector via the function name, so that  SCNRM2 := sqrt( x**H*x )
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \return euclidean norm
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
float nmblas_scnrm2(
	const int N,
	const float* X,
	const int INCX
);
//! \}

	/**
	 *  \brief copies a vector, X, to a vector, Y
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly 
	 */
//! \{
void nmblas_scopy(
	const int N,
	const float* X,
	const int  INCX,
	float* Y,
	const int INCY
);
//! \}

	/**
	 *  \brief SDOT forms the dot product of two vectors.
	 *  \ingroup LEVEL1
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \return Return dot product of two vectors
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly 
	 */
//! \{
float nmblas_sdot(
	const int N,
	const float *X,
	const int INCX,
	const float *Y,
	const int INCY
);
//! \}

	/**
	 *  \brief Compute the inner product of two vectors with double precision accumulation.
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] B single precision scalar to be added to inner product
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX ) ) single precision vector with N elementsr
     *  \param [in] INCX storage spacing between elements of X
	 *  \param [in] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \return Returns S.P. result with dot product accumulated in D.P.
     *  SDSDOT = SB + sum for I = 0 to N-1 of SX(LX+I*INCX)*SY(LY+I*INCY),
     *  where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
     *  defined in a similar way using INCY.
	 *  
	 *  \details 
	 */
//! \{
float nmblas_sdsdot(
	const int N,
	const float B,
	const float* X,
	const int INCX,
	const float* Y,
	const int INCY
);
//! \}

	/**
	 *  \brief SNRM2 returns the euclidean norm of a vector via the function name, so that SNRM2 := sqrt( x'*x ).
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] X array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \return  euclidean norm 
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
float nmblas_snrm2(
	const int N,
	const float* X,
	const int INCX
);
//! \}
	/**
	 *  \brief applies a plane rotation
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in,out] X  array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \param [in] C 
	 *  \param [in] S 
	 *  
	 *  \details 
	 */
//! \{
void nmblas_srot(
	const int N,
	float *X,
	const int INCX,
	float *Y,
	const int INCY,
	const float C,
	const float S
);
//! \}

	/**
	 *  \brief  Apply a Given's rotation constructed by SROTMG.
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in,out] X  array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 *  \param [in] param array, dimension (5). The  rotation  values  constructed  by SROTMG
	 *  
	 *  \details 
	 */
//! \{
void nmblas_srotm(
	const int N,
	float *X,
	const int INCX,
	float *Y,
	const int INCY,
	float *param
);
//! \}
	/**
	 *  \brief scales a vector by a constant
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in] A specifies the scalar alpha.
	 *  \param [in,out] X array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
				 <param name=" A "> 5 </param>
                 <param name=" X "> im1 </param>
	    		 <param name=" INCX "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
void nmblas_sscal(
	const int N,
	const float A,
	const float *X,
	const int INCX
); 
//! \}

	/**
	 *  \brief interchanges two vectors
	 *  \ingroup LEVEL1	 
	 *  
	 *  \param [in] N number of elements in input vector
	 *  \param [in,out] X  array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
	 *  \param [in] INCX storage spacing between elements of X
	 *  \param [in,out] Y array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
	 *  \param [in] INCY storage spacing between elements of Y
	 	\par
        \xmlonly
            <testperf>
	    	     <param name=" N "> 8 128 1024 2048 </param>
                 <param name=" X "> im1 im2 </param>
	    		 <param name=" INCX "> 1 </param>
				 <param name=" Y "> im1 im2 </param>
				 <param name=" INCY "> 1 </param>
				 <size> N </size>
            </testperf>
		\endxmlonly
	 */
//! \{
void nmblas_sswap(
	const int N,
	const float *X,
	const int INCX,
	const float *Y,
	const int INCY
);
//! \}

/////////////////LEVEL 2 

//defgroup sgemv nmblas_sgemv
   /**
	*  
	*  \ingroup LEVEL2
    *  \brief performs one of the matrix-vector operations   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y, where alpha and beta are scalars, x and y are vectors and A is an  m by n matrix.
    *  
    *  \param [in] TRANS is CHARACTER*1
           On entry, TRANS specifies the operation to be performed as
           follows:

              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

              TRANS = 'T' or 't'   y := alpha*A**T*x + beta*y.

              TRANS = 'C' or 'c'   y := alpha*A**T*x + beta*y.
    *  \param [in] M specifies the number of rows of the matrix A. M must be at least zero.
    *  \param [in] N specifies the number of columns of the matrix A. N must be at least zero.
    *  \param [in] ALPHA LPHA specifies the scalar alpha
    *  \param [in] A array, dimension ( LDA, N ) Before entry, the leading m by n part of the array A must           contain the matrix of coefficients
    *  \param [in] LDA specifies the first dimension of A as declared in the calling (sub) program. LDA must be at least           max( 1, m ).
    *  \param [in] X array, dimension at least
           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
           and at least
           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
           Before entry, the incremented array X must contain the
           vector x.
    *  \param [in] INCX INCX specifies the increment for the elements of X. INCX must not be zero.
    *  \param [in] BETA  specifies the scalar beta. When BETA is supplied as zero then Y need not be set on input. 
    *  \param [in,out] Y is REAL array, dimension at least
           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
           and at least
           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
           Before entry with BETA non-zero, the incremented array Y
           must contain the vector y. On exit, Y is overwritten by the
           updated vector y.
    *  \param [in] INCY specifies the increment for the elements of Y. INCY must not be zero.
    *  \return 
    *  
    *  \details 
    */
//! \{
void nmblas_sgemv(
   const enum nm_trans        TRANS,
   const int                  M,
   const int                  N,
   const float               ALPHA,
   const float               * A,
   const int                  LDA,
   const float               * X,
   const int                  INCX,
   const float               BETA,
   float                     * Y,
   const int                  INCY
);
//! \}

//defgroup dgemv nmblas_dgemv
   /**
	*  
	*  \ingroup LEVEL2
    *  \brief performs one of the matrix-vector operations   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y, where alpha and beta are scalars, x and y are vectors and A is an  m by n matrix.
    *  
    *  \param [in] TRANS is CHARACTER*1
           On entry, TRANS specifies the operation to be performed as
           follows:

              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

              TRANS = 'T' or 't'   y := alpha*A**T*x + beta*y.

              TRANS = 'C' or 'c'   y := alpha*A**T*x + beta*y.
    *  \param [in] M specifies the number of rows of the matrix A. M must be at least zero.
    *  \param [in] N specifies the number of columns of the matrix A. N must be at least zero.
    *  \param [in] ALPHA LPHA specifies the scalar alpha
    *  \param [in] A array, dimension ( LDA, N ) Before entry, the leading m by n part of the array A must           contain the matrix of coefficients
    *  \param [in] LDA specifies the first dimension of A as declared in the calling (sub) program. LDA must be at least           max( 1, m ).
    *  \param [in] X array, dimension at least
           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
           and at least
           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
           Before entry, the incremented array X must contain the
           vector x.
    *  \param [in] INCX INCX specifies the increment for the elements of X. INCX must not be zero.
    *  \param [in] BETA  specifies the scalar beta. When BETA is supplied as zero then Y need not be set on input. 
    *  \param [in,out] Y is REAL array, dimension at least
           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
           and at least
           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
           Before entry with BETA non-zero, the incremented array Y
           must contain the vector y. On exit, Y is overwritten by the
           updated vector y.
    *  \param [in] INCY specifies the increment for the elements of Y. INCY must not be zero.
    *  \return 
    *  
    *  \details 
    */
//! \{
void nmblas_dgemv(
   const enum nm_trans        TRANS,
   const int                  M,
   const int                  N,
   const double               ALPHA,
   const double               * A,
   const int                  LDA,
   const double               * X,
   const int                  INCX,
   const double               BETA,
   double                     * Y,
   const int                  INCY
);
//! \}

//defgroup sger nmblas_sger
   /**
	*  
	*  \ingroup LEVEL2
    *  \brief perform the rank 1 operation   A := alpha*x*y' + A, where alpha is a scalar, x is an m element vector, y is an n element vector and A is an m by n matrix.
    *  
    *  \param [in] M  M is INTEGER
           On entry, M specifies the number of rows of the matrix A.
           M must be at least zero.
    *  \param [in] N  N is INTEGER
           On entry, N specifies the number of columns of the matrix A.
           N must be at least zero.
    *  \param [in] ALPHA ALPHA is REAL
           On entry, ALPHA specifies the scalar alpha.
    *  \param [in] X X is REAL array, dimension at least
           ( 1 + ( m - 1 )*abs( INCX ) ).
           Before entry, the incremented array X must contain the m
           element vector x.
    *  \param [in] INCX  INCX is INTEGER
           On entry, INCX specifies the increment for the elements of
           X. INCX must not be zero.
    *  \param [in] Y Y is REAL array, dimension at least
           ( 1 + ( n - 1 )*abs( INCY ) ).
           Before entry, the incremented array Y must contain the n
           element vector y.
    *  \param [in] INCY INCY is INTEGER
           On entry, INCY specifies the increment for the elements of
           Y. INCY must not be zero.
    *  \param [in] INCX INCX specifies the increment for the elements of X. INCX must not be zero.
    *  \param [in,out] A   A is REAL array, dimension ( LDA, N )
           Before entry, the leading m by n part of the array A must
           contain the matrix of coefficients. On exit, A is
           overwritten by the updated matrix. 
    *  \param [in] LDA   LDA is INTEGER
           On entry, LDA specifies the first dimension of A as declared
           in the calling (sub) program. LDA must be at least
           max( 1, m ).
    *  
    *  \return 
    *  
    *  \details 
    */
//! \{
void nmblas_sger( 
	const int                  M,//row
   const int                  N,//coullum
   const float                ALPHA,
   const float                * X,
   const int                  INCX,
   const float                * Y,
   const int                  INCY,
   float                      * A,
   const int                  LDA
);
//! \}
#include<nmblas/nmblas_sgemm.h>

void MullMatrix_f( 
	void* 	A, 
	int 	pI, 
	int 	ldA, 
	void* 	B,
    int 	pK, 
    int 	ldB,
    void* 	C,
    int 	pJ, 
    int 	ldC,
    int 	plusC 
);


#ifdef __cplusplus
		};
#endif
	
#endif // _INIT_H_INCLUDED_