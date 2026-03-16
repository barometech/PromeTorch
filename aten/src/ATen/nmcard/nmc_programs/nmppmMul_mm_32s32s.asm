//------------------------------------------------------------------------
//
//  $Workfile:: mtrMul_M32M32.as $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.1 $      $Date: 2005/01/12 14:05:15 $
//
//! \if file_doc
//!
//! \file   mtrMul_M32M32.asm
//! \author Сергей Мушкаев
//! \brief  Умножение матрицы на матрицу.
//!
//! \endif
//!
//------------------------------------------------------------------------


begin ".text_nmplm"
/////////////////////////////////////////////////////////////////////////////////////////
	

//--------------------------------------------------------------------
//! \fn void nmppmMul_mm_32s32s( nm32s* pSrcMtr1, int nHeight1, int nWidth1,  nm32s* pSrcMtr2, nm32s* pDstMtr, int nWidth2);
//!
//! \perfinclude nmppmMul_mm_32s32s.html
//--------------------------------------------------------------------
extern mtrMul_mm_32sXs:label;
extern _nmppsTmpBuffer64_G_: long[64];
const LongColumns=_nmppsTmpBuffer64_G_;


global _nmppmMul_mm_32s32s:label;
<_nmppmMul_mm_32s32s>
.branch;

 
	ar5 = sp-2;
	push ar0,gr0;
	push ar1,gr1;
	push ar2,gr2;
	push ar3,gr3;
	push ar4,gr4;
	push ar5,gr5;
	push ar6,gr6;

	//-----------------------------------------
	ar0 = [--ar5];			// SrcMatrixA
	gr5 = [--ar5];			// nHeight A
	gr0 = [--ar5];			// nWidth  A in 32-bit words
	ar2 = [--ar5];			// SrcMatrixB
	ar6 = [--ar5];			// DstMatrix
	gr4 = [--ar5];			// nWidth  B in 32-bit words
	gr7 = gr4>>1;
	[LongColumns] = gr7;
	nb1 = 80000000h with gr6 = gr4;	
	
	<Next_MulMV>
		push ar0,gr0;
		push ar6,gr6;
		
		delayed call mtrMul_mm_32sXs;
			ar4 = ar2;//nul;
			nul;

		pop ar6,gr6;
		pop ar0,gr0;
		gr7 = [LongColumns];
		ar2+=2;
		ar6+=2 with gr7--;
	if <>0 delayed goto Next_MulMV;
		[LongColumns] = gr7;

	pop ar6,gr6;
	pop ar5,gr5;
	pop ar4,gr4;
	pop ar3,gr3;
	pop ar2,gr2;
	pop ar1,gr1;
	pop ar0,gr0;


return with gr7=false;
.wait;



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Low level function of multyplying 32-bit matrix(SrcMatrix) by matrix(pSrcVec)
// with user-defined bit capacity and summary bit width of row = 64 bits.
// User-defined matrix(pSrcVec) and result product(pDstVec) are represented 
// as vectors of 64-bit packed words(long), partition of wich is defined by gr3(nb1) register.
//
// INPUT REGISTERS:
// ar0->SrcMatrix											:long Global[nHeight*Width/2]
// gr0= SrcMatrix nWidth in 32-bit words						:=[2,4,6....]
// gr5= SrcMatrix nHeight									:=[0,1,2,3..]
//
// ar4->pSrcVec											:long Local [nWidth*gr4/2]
// gr4= pSrcVec long to long reading step in 32-bit words :=[0,2,4,6....]
//
// ar6->pDstVec											:long Global[nHeight*gr6/2]
// gr6= pDstVec long to long writing step in 32-bit words	:=[0,2,4,6....]
//
// gr3= nb1
// COMMENTS: gr4 and gr6 registers are need for vector extraction and placing directly from/into matrixes
// 
// NOTE:  Registers : ar0,ar1,   ,ar3,ar4,ar5,ar6,   ,
//                   ,   ,gr1,gr2,   ,   ,   ,   ,gr7   will be changed after return !


end ".text_nmplm"; 
