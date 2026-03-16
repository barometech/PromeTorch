/* Minimal startup code for NMC4 */

	.section .text.init, "ax"
	.globl start
	.type start, @function

start:
	/* Set stack pointer to end of NMMB */
	ar7 = 0x7F000;

	/* Clear frame pointer */
	ar6 = 0;

	/* Call __main (C++ mangled name) */
	call __main;

	/* If main returns, halt in infinite loop */
_halt:
	delayed goto _halt;

	.size start, .-start

/* Provide _exit stub */
	.globl _exit
	.type _exit, @function
_exit:
	delayed goto _halt;
	.size _exit, .-_exit

/* BSS table markers */
	.section .bss_table, "a"
	.globl __bss_table_start
	.globl __bss_table_end
__bss_table_start:
__bss_table_end:
