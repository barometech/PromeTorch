//----------------------------------------------------------------------------//
//                                                                            //
//                           NM6408 board examples                            //
//                                                                            //
//                                Demo program                                //
//                         calculate factorial of 10.                         //
//                                                                            //
//----------------------------------------------------------------------------//

int fact(int n)
{
	if (n == 1)
		return 1;

	int res = fact(n - 1) * n;

	return res;
}

int main()
{
	int res = fact(10);

	return res;
}
