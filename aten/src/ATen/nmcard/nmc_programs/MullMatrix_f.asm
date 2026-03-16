
global _MullMatrix_f : label;
global _nmppmMul_mm_32f : label;

begin ".text_nmblas"


    
PlusC : label;

<_MullMatrix_f>
<_nmppmMul_mm_32f>

	push ar6, gr6;
	ar6 = ar7;
	push ar0,gr0;
	push ar1,gr1;
	push ar2,gr2;
	push ar3,gr3;
	push ar4,gr4;
	push ar5,gr5;

    
	ar2  = [ar6-=5];                                 //        A
    ar0  = ar2;
	gr7  = [--ar6];                                  //        I
	gr0  = [--ar6];                                  //        ldA
	ar5  = [--ar6] with gr7--;                       //        B
	gr3  = [--ar6] with gr6 = gr7 >> 5;              //        K
	gr1  = [--ar6] with gr3 >>=1;                                  //        ldB
    ar1  = ar5     with gr4 = false;
    ar3  = gr4;
    
    ar4  = [--ar6];                                  //  C
    gr4  = [--ar6];                                  //  J
    
    gr2  = [ar6-2] with gr1 <<=1;        //  bool : A*B ? A*B+C
    vlen = gr7 with gr5 = gr3;    

	push ar5,gr5;                          //          SAVE B,K        push 1
	fpu 0  rep vlen   vreg6=   [ar0++gr0];
	push ar0,gr0;                          //          NEXT_A (I+VLEN) push 1
    
<I_J_Loop>
	fpu 1  vreg6 = fpu 0 vreg6;
	fpu 2  vreg6 = fpu 1 vreg6;
	fpu 3  vreg6 = fpu 2 vreg6;

	fpu 0  rep  4   vreg0=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 1  rep  4   vreg0=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 2  rep  4   vreg0=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 3  rep  4   vreg0=   [ar1++gr1];
    ar5 -= 6 with gr5 = gr1 >> 1;
    ar5 += gr5;
    ar1  = ar5;
	fpu 0  rep  4   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 1  rep  4   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 2  rep  4   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr5 =-gr5;
	fpu 3  rep  4   vreg1=   [ar1++gr1];
    ar5 += gr5;
    ar5 -= 6 with gr5 = gr1 << 2;
    ar5 += gr5 with gr2;             //  flags <- A*B ? A*B+C
    ar1  = ar5;
    
    if <>0 delayed goto PlusC with gr3--;              //  jump  <- A*B ? A*B+C
    ar2 += 2;
    
    ar0  = ar2;
	fpu 0  rep vlen   vreg7=   [ar0++gr0];
	fpu 1  vreg7 = fpu 0 vreg7;
	fpu 2  vreg7 = fpu 1 vreg7;
	fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 );
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 );
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 );
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 );
    
<fromPlusC>

    if =0 delayed goto MainLoopEnd;
    ar2 += 2;
    ar0  = ar2;
	fpu 0  rep vlen   vreg3=   [ar0++gr0];
	fpu 1  vreg3 = fpu 0 vreg3;
	fpu 2  vreg3 = fpu 1 vreg3;
	fpu 3  vreg3 = fpu 2 vreg3;
    
	fpu 0  rep  9   vreg4=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 1  rep  9   vreg4=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;

    ar2 += 2;
    ar0  = ar2;
	fpu 0  rep vlen   vreg6=   [ar0++gr0];
	fpu 1  vreg6 = fpu 0 vreg6;
	fpu 2  vreg6 = fpu 1 vreg6;
	fpu 3  vreg6 = fpu 2 vreg6;

	fpu 2  rep  9   vreg4=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 3  rep  9   vreg4=   [ar1++gr1];
    ar5 -= 6 with gr5 = gr1 >> 1;
    ar5 += gr5;
    ar1  = ar5 with gr3--;
    
	fpu 0  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    
	fpu 1  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5;
	fpu 2  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2 with gr5 =-gr5;
    ar1  = ar5;
	fpu 3  rep  9   vreg5=   [ar1++gr1];
    ar5 += gr5 with gr5 = gr1 << 3;
    ar5 -= 6 with gr5 += gr1;
    ar5 += gr5;
    ar1  = ar5;
    
    ar2 += 2;
    ar0  = ar2 with gr3--;
	fpu 0  rep vlen   vreg7=   [ar0++gr0];
	fpu 1  vreg7 = fpu 0 vreg7;
	fpu 2  vreg7 = fpu 1 vreg7;
	fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;

/////////////////////////
/////////////////////////
    
    ar2 += 2;
<MainLoopA>     //  using ( vreg4,vreg5 ) loading ( vreg0,vreg1 )
    ar0  = ar2;
    fpu 0  rep vlen   vreg3=   [ar0++gr0];
    fpu 1  vreg3 = fpu 0 vreg3;
    fpu 2  vreg3 = fpu 1 vreg3;
    fpu 3  vreg3 = fpu 2 vreg3;
    
    fpu 0  rep  9   vreg0=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg6=   [ar0++gr0];
    fpu 1  vreg6 = fpu 0 vreg6;
    fpu 2  vreg6 = fpu 1 vreg6;
    fpu 3  vreg6 = fpu 2 vreg6;
    
    fpu 1  rep  9   vreg0=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg7=   [ar0++gr0];
    fpu 1  vreg7 = fpu 0 vreg7;
    fpu 2  vreg7 = fpu 1 vreg7;
    fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 2  rep  9   vreg0=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg3=   [ar0++gr0];
    fpu 1  vreg3 = fpu 0 vreg3;
    fpu 2  vreg3 = fpu 1 vreg3;
    fpu 3  vreg3 = fpu 2 vreg3;
    
    fpu 3  rep  9   vreg0=   [ar1++gr1];
    ar5 -= 6 with gr5 = gr1 >> 1;
    ar5 += gr5;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg6=   [ar0++gr0];
    fpu 1  vreg6 = fpu 0 vreg6;
    fpu 2  vreg6 = fpu 1 vreg6;
    fpu 3  vreg6 = fpu 2 vreg6;
    
    fpu 0  rep  9   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg7=   [ar0++gr0];
    fpu 1  vreg7 = fpu 0 vreg7;
    fpu 2  vreg7 = fpu 1 vreg7;
    fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 1  rep  9   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg3=   [ar0++gr0];
    fpu 1  vreg3 = fpu 0 vreg3;
    fpu 2  vreg3 = fpu 1 vreg3;
    fpu 3  vreg3 = fpu 2 vreg3;
    
    fpu 2  rep  9   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg6=   [ar0++gr0];
    fpu 1  vreg6 = fpu 0 vreg6;
    fpu 2  vreg6 = fpu 1 vreg6;
    fpu 3  vreg6 = fpu 2 vreg6;
    
    fpu 3  rep  9   vreg1=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg7=   [ar0++gr0];
    fpu 1  vreg7 = fpu 0 vreg7;
    fpu 2  vreg7 = fpu 1 vreg7;
    fpu 3  vreg7 = fpu 2 vreg7;
    
    //-- no load B here --
    gr5 =-gr5;
    ar5 += gr5 with gr5 = gr1 << 3;
    ar5 -= 8 with gr5 += gr1;
    ar5 += gr5;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg4,vreg5 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg3=   [ar0++gr0];
    fpu 1  vreg3 = fpu 0 vreg3;
    fpu 2  vreg3 = fpu 1 vreg3;
    fpu 3  vreg3 = fpu 2 vreg3;
    
    fpu 0  rep  9   vreg4=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg6=   [ar0++gr0];
    fpu 1  vreg6 = fpu 0 vreg6;
    fpu 2  vreg6 = fpu 1 vreg6;
    fpu 3  vreg6 = fpu 2 vreg6;
    
    fpu 1  rep  9   vreg4=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg7=   [ar0++gr0];
    fpu 1  vreg7 = fpu 0 vreg7;
    fpu 2  vreg7 = fpu 1 vreg7;
    fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 2  rep  9   vreg4=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg3=   [ar0++gr0];
    fpu 1  vreg3 = fpu 0 vreg3;
    fpu 2  vreg3 = fpu 1 vreg3;
    fpu 3  vreg3 = fpu 2 vreg3;
    
    fpu 3  rep  9   vreg4=   [ar1++gr1];
    ar5 -= 6 with gr5 = gr1 >> 1;
    ar5 += gr5;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg6=   [ar0++gr0];
    fpu 1  vreg6 = fpu 0 vreg6;
    fpu 2  vreg6 = fpu 1 vreg6;
    fpu 3  vreg6 = fpu 2 vreg6;
    
    fpu 0  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg7=   [ar0++gr0];
    fpu 1  vreg7 = fpu 0 vreg7;
    fpu 2  vreg7 = fpu 1 vreg7;
    fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 1  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg3=   [ar0++gr0];
    fpu 1  vreg3 = fpu 0 vreg3;
    fpu 2  vreg3 = fpu 1 vreg3;
    fpu 3  vreg3 = fpu 2 vreg3;
    
    fpu 2  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg7 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg6=   [ar0++gr0];
    fpu 1  vreg6 = fpu 0 vreg6;
    fpu 2  vreg6 = fpu 1 vreg6;
    fpu 3  vreg6 = fpu 2 vreg6;
    
    fpu 3  rep  9   vreg5=   [ar1++gr1];
    ar5 += 2;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg3 * .retrive( vreg0,vreg1 ) +vreg2;
    
    ar2 += 2;
    ar0  = ar2;
    fpu 0  rep vlen   vreg7=   [ar0++gr0];
    fpu 1  vreg7 = fpu 0 vreg7;
    fpu 2  vreg7 = fpu 1 vreg7;
    fpu 3  vreg7 = fpu 2 vreg7;
    
    //-- no load B here --
    gr5 =-gr5;
    ar5 += gr5 with gr5 = gr1 << 3;
    ar5 -= 8 with gr5 += gr1;
    ar5 += gr5;
    ar1  = ar5 with gr3--;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    if =0 delayed goto MainLoopEnd;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) +vreg2;
/////////////////////////
/////////////////////////
    
    delayed goto MainLoopA;
    ar2 += 2;
    
<MainLoopEnd>
	gr0 = [--ar6] with gr4>>=1;                     //  ldC
    ar0 = ar4;
    
    fpu 0  rep vlen   [ar0++gr0]= vreg2;    
    gr5  = ar3;
    with gr5;
    if <>0 goto KeepAR3 with gr4--;
    
    ar3  = ar0;
<KeepAR3>
    if =0 delayed goto UnloadEnd with gr6;
    ar4 += 2;
    ar0  = ar4 with gr4--;
    fpu 1  rep vlen   [ar0++gr0]= vreg2;    
    if =0 delayed goto UnloadEnd with gr6;
    ar4 += 2;
    ar0  = ar4 with gr4--;
    fpu 2  rep vlen   [ar0++gr0]= vreg2;    
    if =0 delayed goto UnloadEnd with gr6;
    ar4 += 2;
    ar0  = ar4 with gr4--;
    fpu 3  rep vlen   [ar0++gr0]= vreg2;    
    if <>0 delayed goto Jcycle with gr4 <<=1;
    ar4 += 2 with gr6;                          //  I

<UnloadEnd>
    if <>0 delayed goto Icycle with gr6--;
    vlen = 31;

    ar7 -=4;
	pop ar5,gr5;
	pop ar4,gr4;
	pop ar3,gr3;
	pop ar2,gr2;
	pop ar1,gr1;
	pop ar0,gr0;
	pop ar6,gr6;
    
return;                                  //<51,1>

<Icycle>
    pop ar0, gr0;                           //          NEXT_A (I+VLEN)  pop
    pop ar5, gr5;                           //          RESTORE B, K     pop
    push ar5, gr5;
    ar2  = ar0;
    [ar6 +=8]= ar2;                         //          setup A for J cycle
    [ar6 -=3]= ar5 with gr3 = gr5;          //          setup B for J cycle
    ar1  = ar5;
    ar4  = ar3 with gr4= false;                             //          C   
    ar3  = gr4;
    gr4  = [ar6-=4];                        //          restore J

    delayed goto I_J_Loop;
	fpu 0  rep vlen   vreg6=   [ar0++gr0];
	push ar0,gr0;                           //          NEXT_A (I+VLEN) push 2
/**/
<Jcycle>
	ar2  = [ar6+=8];                        //        A
	ar6--;                                  //        I   (vlen)
	gr0  = [--ar6];                         //        ldA
	ar5  = [--ar6];                         //        B
    ar5  = ar5 + 8;
    [ar6]= ar5;                             //        B (new)
	gr3  = [--ar6];                         //        K
	ar6-=3 with gr3 >>=1;                    //        ldB  (gr1)

    ar0  = ar2;
    delayed goto I_J_Loop;
    ar1  = ar5;
	fpu 0  rep vlen   vreg6=   [ar0++gr0];



<PlusC>             //  keep flags!
    push ar0,gr0;
    
	gr0 = [--ar6];                                  //<40,27> ldC
    ar6++;
    ar0 = ar4;

    fpu 0  rep vlen   vreg2= [ar0++gr0];
    ar4 += 2;
    ar0  = ar4;
    fpu 1  rep vlen   vreg2= [ar0++gr0];
    ar4 += 2;
    ar0  = ar4;
    fpu 2  rep vlen   vreg2= [ar0++gr0];
    ar0  = ar4 + 2;
    ar4 -= 4;
    fpu 3  rep vlen   vreg2= [ar0++gr0];

    pop ar0,gr0;
    
    ar0  = ar2;
	fpu 0  rep vlen   vreg7=   [ar0++gr0];
	fpu 1  vreg7 = fpu 0 vreg7;
	fpu 2  vreg7 = fpu 1 vreg7;
	fpu 3  vreg7 = fpu 2 vreg7;
    
    fpu 0  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) + vreg2;
    fpu 1  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) + vreg2;
delayed goto fromPlusC;
    fpu 2  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) + vreg2;
    fpu 3  .matrix  vreg2=   vreg6 * .retrive( vreg0,vreg1 ) + vreg2;
    


end ".text_nmblas";


