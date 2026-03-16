/* Linker script for NM Card Mini - with DDR section */

MEMORY
{
    /* On-chip SRAM: 512KB */
    NMMB : ORIGIN = 0x00000800, LENGTH = 0x0007F800

    /* DDR3L: доступ через EMI, ~500MB для данных */
    EMI_CPU : ORIGIN = 0x00340000, LENGTH = 0x1FCC0000
}

ENTRY(start)

SECTIONS
{
    /* Force start address to 0x800 */
    . = 0x800;

    /* Startup code MUST be first */
    .text.init : ALIGN(0x8)
    {
        *(.text.init);
    } > NMMB

    .text : ALIGN(0x8)
    {
        *(.text);
        *(.text.*);
    } > NMMB

    .init : ALIGN(0x8)
    {
        *(.init);
    } > NMMB

    .fini : ALIGN(0x8)
    {
        *(.fini);
    } > NMMB

    .rodata : ALIGN(0x8)
    {
        *(.rodata);
        *(.rodata.*);
    } > NMMB

    .data : ALIGN(0x8)
    {
        *(.data);
    } > NMMB

    /* DDR section for large buffers */
    .data.ddr : ALIGN(0x8)
    {
        *(.data.ddr);
    } > EMI_CPU

    .bss : ALIGN(0x8)
    {
        *(.bss);
    } > NMMB

    .bss.ddr : ALIGN(0x8)
    {
        *(.bss.ddr);
    } > EMI_CPU

    .stack : ALIGN(0x8)
    {
        *(.stack);
    } > NMMB
}
