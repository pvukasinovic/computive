/*
 * interrupt.c -- GIC interrupt setup for the ML accelerator
 *
 * Target: Kria KV260 (Zynq UltraScale+ MPSoC, ARM Cortex-A53)
 *
 * UltraScale+ specifics:
 *   - GIC is GICv2 (XScuGic driver from Xilinx standalone BSP)
 *   - PL-to-PS interrupts (IRQ_F2P) map to GIC SPI IDs 121-128
 *     (NOT 61-68 as on Zynq-7000)
 *   - IRQ_F2P[0] = SPI ID 121
 *   - Trigger type MUST be level-sensitive (0x1) for UltraScale+ PL IRQs.
 *     Edge-triggered (0x3) will cause missed or spurious interrupts because
 *     the PL interrupt output is active-high level, not pulsed.
 *   - Timer frequency: 750 MHz (CPU_FREQ / 2 of 1.5 GHz A53 cluster)
 *
 * SPDX-License-Identifier: MIT
 */

#include "xscugic.h"
#include "xil_exception.h"
#include "accel.h"

/* ---------------------------------------------------------------------------
 * GIC IRQ identifiers
 * ---------------------------------------------------------------------------*/

/* Accelerator "done/error" interrupt: PL IRQ_F2P[0] = GIC SPI 121 */
#define ACCEL_IRQ_ID       121

/* DMA interrupt IDs -- these depend on the Vivado block design.
 * Typical mapping for the first AXI DMA instance:
 *   MM2S interrupt -> IRQ_F2P[1] = SPI 122
 *   S2MM interrupt -> IRQ_F2P[2] = SPI 123
 * Adjust if your block design routes them differently.                      */
#define DMA_MM2S_IRQ_ID    122
#define DMA_S2MM_IRQ_ID    123

#define INTC_DEVICE_ID     XPAR_SCUGIC_SINGLE_DEVICE_ID

/* Priority: 0x00 = highest, 0xF8 = lowest (5-bit field, bits [7:3]).
 * 0xA0 is a sensible medium priority that leaves room above and below.      */
#define IRQ_PRIORITY       0xA0

/* Trigger type:
 *   0x1 = level-sensitive (REQUIRED for UltraScale+ PL interrupts)
 *   0x3 = rising-edge (do NOT use for PL IRQs on UltraScale+)              */
#define IRQ_TRIGGER_LEVEL  0x1

/* ---------------------------------------------------------------------------
 * Static GIC instance
 * ---------------------------------------------------------------------------*/
static XScuGic g_intc;

/* ---------------------------------------------------------------------------
 * DMA ISR stubs
 *
 * In the polled driver (accel_run_inference) these are not strictly needed
 * because the driver polls DMA status registers directly.  They are provided
 * so the interrupt-driven path can optionally use DMA interrupts instead of
 * polling DMA completion.  For now they just clear the DMA interrupt flags.
 * ---------------------------------------------------------------------------*/
static void dma_mm2s_isr(void *data)
{
    (void)data;
    /* Read MM2S status and clear interrupt flags (W1C) */
    uint32_t sr = REG_READ(DMA_BASE, DMA_MM2S_DMASR);
    REG_WRITE(DMA_BASE, DMA_MM2S_DMASR, sr & (DMA_SR_IOC_IRQ | DMA_SR_ERR_IRQ));
}

static void dma_s2mm_isr(void *data)
{
    (void)data;
    /* Read S2MM status and clear interrupt flags (W1C) */
    uint32_t sr = REG_READ(DMA_BASE, DMA_S2MM_DMASR);
    REG_WRITE(DMA_BASE, DMA_S2MM_DMASR, sr & (DMA_SR_IOC_IRQ | DMA_SR_ERR_IRQ));
}

/* ---------------------------------------------------------------------------
 * setup_interrupts -- Initialize GIC and connect all PL interrupt handlers
 *
 * Returns 0 on success, -1 on any failure.
 * ---------------------------------------------------------------------------*/
int setup_interrupts(void)
{
    XScuGic_Config *cfg;
    int status;

    /* ---- Initialize GIC ------------------------------------------------ */
    cfg = XScuGic_LookupConfig(INTC_DEVICE_ID);
    if (!cfg)
        return -1;

    status = XScuGic_CfgInitialize(&g_intc, cfg, cfg->CpuBaseAddress);
    if (status != XST_SUCCESS)
        return -1;

    /* ---- Connect to ARM exception system ------------------------------- */
    Xil_ExceptionRegisterHandler(
        XIL_EXCEPTION_ID_INT,
        (Xil_ExceptionHandler)XScuGic_InterruptHandler,
        &g_intc);

    /* ---- Accelerator interrupt (DONE / ERROR) -------------------------- */
    status = XScuGic_Connect(&g_intc, ACCEL_IRQ_ID,
                             (Xil_InterruptHandler)accel_isr,
                             NULL);
    if (status != XST_SUCCESS)
        return -1;

    /* Level-sensitive trigger (0x1) -- critical for UltraScale+ PL IRQs */
    XScuGic_SetPriorityTriggerType(&g_intc, ACCEL_IRQ_ID,
                                   IRQ_PRIORITY, IRQ_TRIGGER_LEVEL);
    XScuGic_Enable(&g_intc, ACCEL_IRQ_ID);

    /* ---- DMA MM2S interrupt -------------------------------------------- */
    status = XScuGic_Connect(&g_intc, DMA_MM2S_IRQ_ID,
                             (Xil_InterruptHandler)dma_mm2s_isr,
                             NULL);
    if (status != XST_SUCCESS)
        return -1;

    XScuGic_SetPriorityTriggerType(&g_intc, DMA_MM2S_IRQ_ID,
                                   IRQ_PRIORITY, IRQ_TRIGGER_LEVEL);
    XScuGic_Enable(&g_intc, DMA_MM2S_IRQ_ID);

    /* ---- DMA S2MM interrupt -------------------------------------------- */
    status = XScuGic_Connect(&g_intc, DMA_S2MM_IRQ_ID,
                             (Xil_InterruptHandler)dma_s2mm_isr,
                             NULL);
    if (status != XST_SUCCESS)
        return -1;

    XScuGic_SetPriorityTriggerType(&g_intc, DMA_S2MM_IRQ_ID,
                                   IRQ_PRIORITY, IRQ_TRIGGER_LEVEL);
    XScuGic_Enable(&g_intc, DMA_S2MM_IRQ_ID);

    /* ---- Enable ARM IRQ exceptions globally ---------------------------- */
    Xil_ExceptionEnable();

    return 0;
}
