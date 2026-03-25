
#ifndef INCLUDED_nmplm
#define INCLUDED_nmplm

#include "nmtype.h"
#include "nmdef.h"
#include "nmpp.h"


/*! \ru
	\page intro Введение
	\en
	\page intro Introduction
	\~
	\ingroup MTR_Group_
	\ru
Назначением данной библиотеки является предоставление базовых операций по обработке матриц для процессорах NM6403, NM6404, NM6405. В состав библиотеки входят арифметические операции над матрицами. Библиотека предназначена для быстрой разработки эффективных пользовательских программ как на языке высокого уровня(С++).

Функции библиотеки имеют C++ интерфейс. Большинство функций библиотеки реализованы на языке ассемблера с использованием векторных инструкций и оптимизированы под архитектуру процессора NM6403.

Для удобства разработки прикладных программ библиотека содержит аналогичные реализации функций для процессоров серии x86, выполненных на языке С++. Данные реализации позволяют выполнять написанные с использованием данной библиотеки прикладные программы на персональном компьютере.
	\en
The purpose of the library is to support the base operations with matrixes on NM6403, NM6404, NM6405 processors. The library сontains arithmetical functions. The library is meant for fast development of effective user programs in a high level language(С++) .

The library functions have a C++ interface. Most of the library functions are implemented in assembly language with using vector core instructions and optimized for the processor architecture.

For convenience of application programs development the library has implementations of functions for processors of x86 series in C++. These implementations allow to execute application programs written by means of this library on a personal computer.
	\~
*/



//*****************************************************************************

	/**
    *
    * \ru
    *     \defgroup mSupport Функции поддержки
    * \en
    *     \defgroup mSupport Support functions
    * \~
    * \ingroup MTR_Group_
	*/
#include "mSupport.h"

//*****************************************************************************
    //--------------------------------------------------------------------
    //! \ru
    //!     \defgroup mInit Инициализация и копирование
    //! \en
    //!     \defgroup mInit Initialization and copy
	//! \~
    //! \ingroup MTR_Group_
    //--------------------------------------------------------------------

#include "mInit.h"

//*****************************************************************************

	/**
    *
    * \ru
    *     \defgroup mMatrixVector Векторно-матричные операции
    * \en
    *     \defgroup mMatrixVector Vector-mtr operations.
    * \~
    * \ingroup MTR_Group_
	*/
#include "mMatrixVector.h"
#include "mMatrixVectorDev.h"
//*****************************************************************************

/**
*
* \ru
*     \defgroup mInverse Обращение матрицы
* \en
*     \defgroup mInverse Matrix inverse
* \~
*     \ingroup MTR_Group_f
*/
#include "mInverse.h"

#include "mStat.h"

//#include "profile\profdata.h"



#endif
