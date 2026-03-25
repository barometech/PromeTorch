MEMORY
{
	NMMB : ORIGIN = 0x00000800, LENGTH = 0x0007F800
	EMI_CPU : ORIGIN = 0x00340000, LENGTH = 0x1FCC0000
}

SECTIONS
{
	.text : ALIGN(0x8)
	{
		*(.text);
	} > NMMB

	.init : ALIGN(0x8)
	{
		*(.init);
	} > NMMB

	.fini : ALIGN(0x8)
	{
		*(.fini);
	} > NMMB

	.data : ALIGN(0x8)
	{
		*(.data);
	} > NMMB

	.bss : ALIGN(0x8)
	{
		*(.bss);
	} > NMMB

	.stack : ALIGN(0x8)
	{
		*(.stack);
	} > NMMB
}
