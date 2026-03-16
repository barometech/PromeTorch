//------------------------------------------------------------------------
//
//  $Workfile:: mtrMul_MxxV64.as $
//
//  Векторно-матричная библиотека
//
//  Copyright (c) RC Module Inc.
//
//  $Revision: 1.1 $      $Date: 2005/01/12 14:05:15 $
//
//! \if file_doc
//!
//! \file   mtrMul_MxxV64.asm
//! \author Сергей Мушкаев
//! \brief  Умножение матрицы на вектор.
//!
//! \endif
//!
//------------------------------------------------------------------------
import from mulmv.mlb;



begin ".text_nmplm"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Low level function of multiplying 32-bit matrix(SrcMatrix) by matrix(pSrcVec)
// with user-defined bit capacity and summary bit width of row = 64 bits.
// User-defined matrix(pSrcVec) and result product(pDstVec) are represented 
// as vectors of 64-bit packed words(long), partition of which is defined by gr3(nb1) register.
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
// COMMENTS: gr4 and gr6 registers are need for vector extraction and placing directly from/into matrixes
// 
// NOTE:  Registers : ar0,ar1,   ,ar3,ar4,ar5,ar6,   ,
//                   ,   ,gr1,gr2,   ,   ,   ,   ,gr7   will be changed after return !
// wtw must be on return
global  mtrMul_mm_32sXs:label;
<mtrMul_mm_32sXs>
	sb=2;
	REP_N_NUL_MxxV64(2);
.wait;



end ".text_nmplm"; 
