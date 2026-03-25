#ifndef __INTRC_NMC_USER_H__
#define __INTRC_NMC_USER_H__

/*
 * NM core supports 4 interrupts (in decreasing order of priority): 
 *  - non-maskable external, 
 *  - overflow, 
 *  - erroneous instruction, 
 *  - peripheral external, 
 *  - step by step. 
 * The interrupt controller accepts interrupt requests from the periphery, 
 * arbitration and fetching the peripheral request for the NM core.
 * 
 * An external peripheral interrupt can only be processed if it has been 
 * enabled in the interrupt controller and an external peripheral interrupt 
 * is enabled in the core. However, a request from a peripheral device 
 * is always latched into the interrupt controller. Therefore, before 
 * enabling interrupts in the controller, the corresponding request is 
 * cleared so that an outdated, irrelevant request cannot invoke 
 * the interrupt.
 *
 * Before fetching a peripheral interrupt for the kernel, requests are 
 * arbitrated. There are two levels of priority: high and low. By default, 
 * all interrupts have a high priority. Within one priority, the interrupt 
 * with the lower number takes precedence.
 *
 * A table of 32 vectors of interrupt handlers and a table for storing 
 * the argument for them are statically allocated in memory. By default, 
 * all vectors point to ic_default_handler, and all arguments are NULL.
 *
 * All interrupt operations are performed using masks. Masks for controlling 
 * interrupts in the core are preset in ic_core_intr_mask, and the mask for 
 * peripheral interrupts for the interrupt controller can be obtained 
 * by the number of this interrupt using ic_intc_mask()
 *
 * Before calling the interrupt handler, all interrupts in the core 
 * are masked. However, the interrupt handler context cannot be considered 
 * atomic because there is a non-maskable interrupt for the core.
 */

#ifdef __cplusplus
extern "C" {
#endif

enum ic_intr_nums
{
    /* core inner interrupts */
    IC_NMI = 0, // Non-maskable external
    IC_OF,      // Overflow
    IC_EI,      // Erroneous Instruction
    IC_ST,      // Step
    /* peripheral interrupts */
    IC_FPUI0,   // FPU Incorrect Data
    IC_FPUI1,   // FPU Overflow
    IC_FPUI2,   // FPU Underflow
    IC_FPUI3,   // FPU Significance Loss
    IC_FPUI4,   // FPU Data Lost
    IC_FPUI5,   // FPU Wrong Command
    IC_IFE,     // System Integrator Fetching Command
    IC_DAE,     // Bridge SI-AXI External Memory Access
    IC_PWI,     // MPU Write Memory Access
    IC_PRI,     // MPU Read Memory Access
    IC_T0I,     // Timer0
    IC_T1I,     // Timer1
    IC_SCI0,    // System Controller Interrupt 0
    IC_SCI1,    // System Controller Interrupt 1
    IC_SCI2,    // System Controller Interrupt 2
    IC_SCI3,    // System Controller Interrupt 3
    IC_CPO0,    // CP0 Send
    IC_CPI0,    // CP0 Receive
    IC_CPO1,    // CP1 Send
    IC_CPI1,    // CP1 Receive
    IC_CPO2,    // CP2 Send
    IC_CPI2,    // CP2 Receive
    IC_CPO3,    // CP3 Send
    IC_CPI3,    // CP3 Receive
    IC_SCI4,    // System Controller Interrupt 4
    IC_SCI5,    // System Controller Interrupt 5
    IC_SCI6,    // System Controller Interrupt 6
    IC_SCI7,    // System Controller Interrupt 7
};

/* Masks for enabling interrupts in the core */
enum ic_pswr_mask
{
    IC_CORE_OF    = 0x00000100,
    IC_CORE_EI    = 0x00000080,
    IC_CORE_ST    = 0x00000020,
    IC_CORE_PER   = 0x00000040
};
#define IC_CORE_TOTAL 0x000001E0

#define IC_PER_MASK(perif_intr_num) (1 << ((perif_intr_num) - 4))
#define IC_PER_TOTAL 0x0FFFFFFF

/* Get mask of the peripheral interrupt for interrupt controller */
static inline int ic_intc_mask(enum ic_intr_nums intr_num)
{
    return intr_num >= 4 ? IC_PER_MASK(intr_num) : 0;
}

typedef void (ic_handler_t)(int id, void *arg);

/* Stub for not used handlers, does nothing */
extern ic_handler_t ic_default_handler;

/* 
 * Registers a handler for the specified interrupt in the irq handler 
 * vector table, the transition to the handler is created 
 * on the corresponding interrupt vector. Also stores the "arg" pointer 
 * to pass to the handler when it is called.
 *
 * Returns the old value of the pointer from the vector table, which 
 * was replaced by the "handler" value passed to the function.
 */
ic_handler_t* ic_set_handler(int intr_num, ic_handler_t* handler, void* arg);
/*
 * Resets handler for specified interrupt in irq handler vector table 
 * by ic_default_handler.
 *
 * Returns the old value of the pointer from the vector table, which 
 * was replaced by ic_default_handler.
 */
ic_handler_t* ic_clear_handler(int intr_num);

/* Returns the value from the irq handler vector table for the interrupt */
ic_handler_t* ic_get_handler(int intr_num);
void* ic_get_argument(int intr_num);

/* Enables interrupts in core by mask */
void ic_core_intr_enable(int pswr_mask);
/* Disables interrupts in core by mask */
void ic_core_intr_disable(int pswr_mask);
/* Returns the mask of enabled interrupts in the core */
int ic_core_get_enabled(void);


/* 
 * Enables peripheral interrupts by mask. Allows the interrupt controller 
 * to pass selected interrupts, and enables peripheral interrupts in the core.
 * 
 * Before the interrupt is activated, the corresponding possible interrupt 
 * request is cleared. For interrupts that have already been activated, it 
 * does nothing.
 *
 * Returns the mask of enabled peripheral interrupts.
 */
int ic_per_intr_enable(int intc_mask);
/* 
 * Disables peripheral interrupts by mask. Masks the selected interrupts 
 * in the interrupt controller.
 *
 * Returns the mask of enabled peripheral interrupts.
 */
int ic_per_intr_disable(int intc_mask);
/*
 * Disables all peripheral interrupts in the interrupt controller, and 
 * disables peripheral interrupts in the core. 
 */
void ic_per_disable_all(void);
/* Returns the mask of enabled peripheral interrupts */
int ic_per_get_enabled(void);

/* 
 * Sets high priority for peripheral interrupts selected by the mask.
 * Returns the mask of peripheral interrupts with low priority.
 */
int ic_per_set_high_prior(int intc_mask);
/* 
 * Sets low priority for peripheral interrupts selected by the mask.
 * Returns the mask of peripheral interrupts with low priority.
 */
int ic_per_set_low_prior(int intc_mask);
/* Returns the low priority interrupts mask */
int ic_per_get_low_prior(void);

/*
 * Disables all interrups in the core, disables all peripheral interrupts
 * in the interrupt controller. Clears all interrupts requests in the core 
 * and in the interrupt controller. Resets peripheral interrupts priority.
 */
void ic_flush_all(void);
/*
 * Reset peripheral interrupts and clears corresponding requests 
 * in the interrupt controller.
 */
void ic_per_intr_flush(int intc_mask);
/* Disables core interrupts and clears corresponding requests */
void ic_core_intr_flush(int pswr_mask);

/* Returns the peripheral interrupt requests mask */
int ic_per_get_requests(void);
/* Returns the core interrupt requests mask */
int ic_core_get_requests(void);
/* 
 * Returns the number of the interrupt request that was latched as an external 
 * peripheral interrupt to the core. Returns 0 if there are no peripheral 
 * interrupt requests.
 */
int ic_core_get_per_num(void);

static inline unsigned int nmc_irqEnable(void)
{
	unsigned int state;

	asm("%0 = pswr;\n":"=r"(state)::);
	asm("pswr set 0x0C0;\n"::"X"(state):"memory");

	return state;
}

static inline unsigned int nmc_irqDisable(void)
{
	unsigned int state;

	asm("%0 = pswr;\n":"=r"(state)::);
	asm("pswr clear 0x1E0;\n"::"X"(state):"memory");

	return state;
}

static inline unsigned int nmc_irqSetState(unsigned int new_state)
{
	unsigned int old_state;

	asm volatile ("%0 = pswr;\n":"=r"(old_state)::);
	asm volatile ("pswr = %0;\n"::"r"(new_state):"cc");

	return old_state;
}

// Condition flags will remain exactly same as before entering to the ATOMIC BLOCK
#define NMC_ATOMIC_BLOCK() \
	for (unsigned int state = nmc_irqDisable(), flag = 1; \
	     flag; \
	     nmc_irqSetState(state), flag = 0)

void mdelay(int msec);
void udelay(unsigned int us);

#ifdef __cplusplus
};
#endif

#endif // __INTRC_NMC_USER_H__
