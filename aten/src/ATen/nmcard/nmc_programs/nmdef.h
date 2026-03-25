
#ifndef __NMDEF_H
#define __NMDEF_H

#ifdef __NM__
	#ifndef NMDEF
		#define NMDEF
	#endif
#endif

#ifdef NM6403
	#ifndef NMDEF
		#define NMDEF
	#endif
#endif


#ifndef NULL
	#define NULL 0
#endif
/*
#ifndef ADDR
	#ifdef __NM__
		#define ADDR(base,disp)  \
		(	sizeof(*(base)) == 1 ? nmppsAddr_((base),(disp)):	\
			((base)+(disp))								\
		)
	#else  
		#define ADDR(base,disp) ((base)+(disp))
	#endif //NMDEF
#endif //ADDR
*/
#ifndef MIN
#define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifndef ABS
#define ABS(a)    ((a) < (0) ? -(a): (a))
#endif

#ifndef ROUND2INT
#define ROUND2INT(a)  ((a) >= 0 ? (int((a)+0.5)) : (int((a)-0.5)))
#endif

#ifdef __NM__
#define GET_CHAR(pArray,Index,val)			(nmppsGetVal_((pArray),(Index),(val)))
#define GET_SHORT(pArray,Index,val)			(nmppsGetVal_((pArray),(Index),(val)))
#define SET_CHAR(pArray,Index,Char)		(nmppsPut_((pArray),(Index),(Char)))
#define SET_SHORT(pArray,Index,Short)	(nmppsPut_((pArray),(Index),(Short)))
#else
#define GET_CHAR(pArray,Index,val)			((val)=(pArray)[(Index)])
#define GET_SHORT(pArray,Index,val)			((val)=(pArray)[(Index)])
#define SET_CHAR(pArray,Index,Char)		((pArray)[(Index)]=(Char))
#define SET_SHORT(pArray,Index,Short)	((pArray)[(Index)]=(Short))
#endif

#define LONG2INT(size)		((size)<<1)

#define long2INT(size)		((size)<<1)
#define int2INT(size)		((size))
#define short2INT(size)		((size)>>1)
#define char2INT(size)		((size)>>2)


#define LONG2long(size)		((size))
#define LONG2int(size)		((size)<<1)
#define LONG2short(size)	((size)<<2)
#define LONG2char(size)		((size)<<3)

#define long2long(size)		((size))
#define long2int(size)		((size)<<1)
#define long2short(size)	((size)<<2)
#define long2char(size)		((size)<<3)

#define int2long(size)		((size)>>1)
#define int2int(size)		((size))
#define int2short(size)		((size)<<1)
#define int2char(size)		((size)<<2)

#define short2long(size)	((size)>>2)
#define short2int(size)		((size)>>1)
#define short2short(size)	((size))
#define short2char(size)	((size)<<1)

#define char2long(size)		((size)>>3)
#define char2int(size)		((size)>>2)
#define char2short(size)	((size)>>1)
#define char2char(size)		((size))


#endif
