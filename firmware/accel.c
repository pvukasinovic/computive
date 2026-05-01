/*
 * accel.c -- Driver implementation for the ML accelerator IP
 *
 * Target: Kria KV260 (Zynq UltraScale+ MPSoC, ARM Cortex-A53)
 *
 * Implements blocking and interrupt-driven inference using the AXI DMA for
 * data movement and the accelerator CSR for control/status.
 *
 * Critical ordering for each inference:
 *   1. Arm S2MM (receive) -- output path ready before data arrives
 *   2. Write CTRL_START   -- accelerator enters BUSY, waits for input stream
 *   3. Trigger MM2S (send)-- streams input into accelerator
 *   4. Wait for STATUS_DONE (polling or interrupt)
 *   5. Wait for S2MM DMA completion
 *   6. Invalidate cache, copy output from DDR buffer
 *
 * CTRL register bit mapping (from axi_lite_ctrl.sv:154-156):
 *   bit 0 = START       (self-clearing)
 *   bit 1 = SOFT_RST    (self-clearing)
 *   bit 2 = CONTINUOUS  (persistent)
 *   There is NO ABORT bit in the RTL.
 *
 * SPDX-License-Identifier: MIT
 */

#include "accel.h"
#include "accel_regs.h"
#include <string.h>

/* Xilinx standalone BSP -- cache management and lightweight printf.
 * These are provided by the BSP; declare them here so the driver compiles
 * without pulling in the full xil_cache.h header (keeps the dependency
 * explicit).                                                                */
extern void Xil_DCacheFlushRange(uintptr_t addr, uint32_t len);
extern void Xil_DCacheInvalidateRange(uintptr_t addr, uint32_t len);
extern void xil_printf(const char *fmt, ...);

/* -----------------------------------------------------------------------
 * Timeout constants (loop iterations, NOT milliseconds)
 * At ~1 GHz A53 a tight register-read loop does roughly one iteration per
 * 10-50 ns, so 1 000 000 iterations is on the order of 10-50 ms.
 * -----------------------------------------------------------------------*/
#define DMA_RESET_TIMEOUT   10000
#define DMA_XFER_TIMEOUT    1000000
#define ACCEL_RESET_TIMEOUT 10000
#define ACCEL_DONE_TIMEOUT  1000000

/* -----------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/
static volatile bool     g_inference_done  = false;
static volatile bool     g_inference_error = false;
static accel_done_cb_t   g_done_callback   = NULL;

/* =======================================================================
 * DMA helpers
 * =======================================================================*/

/* Reset a single DMA channel.  DMA_CR_RESET is self-clearing; poll until
 * the bit drops.  Returns 0 on success, -1 on timeout.                     */
static int dma_reset_channel(uint32_t cr_offset)
{
    REG_WRITE(DMA_BASE, cr_offset, DMA_CR_RESET);
    for (int i = 0; i < DMA_RESET_TIMEOUT; i++) {
        if (!(REG_READ(DMA_BASE, cr_offset) & DMA_CR_RESET))
            return 0;
    }
    return -1;
}

/* Arm the S2MM (receive) channel.  Must be called BEFORE the accelerator
 * starts producing output so the DMA is ready to accept stream data.
 * This is non-blocking: writing S2MM_LENGTH arms the channel, and the DMA
 * waits for incoming TVALID from the accelerator.                           */
static accel_status_t dma_arm_s2mm(uint32_t len)
{
    /* Invalidate destination cache lines BEFORE DMA writes to them.
     * Cortex-A53 cache line = 64 bytes.                                    */
    Xil_DCacheInvalidateRange(OUTPUT_BUF_ADDR, len);

    REG_WRITE(DMA_BASE, DMA_S2MM_DA, OUTPUT_BUF_ADDR);
    REG_WRITE(DMA_BASE, DMA_S2MM_DMACR, DMA_CR_RS);
    REG_WRITE(DMA_BASE, DMA_S2MM_LENGTH, len);   /* triggers transfer       */

    return ACCEL_OK;
}

/* Wait for S2MM completion.  Returns ACCEL_OK when IOC fires, or
 * ACCEL_ERR_DMA / ACCEL_ERR_TIMEOUT.                                       */
static accel_status_t dma_wait_s2mm(void)
{
    for (uint32_t i = 0; i < DMA_XFER_TIMEOUT; i++) {
        uint32_t sr = REG_READ(DMA_BASE, DMA_S2MM_DMASR);
        if (sr & DMA_SR_ERR_IRQ) {
            REG_WRITE(DMA_BASE, DMA_S2MM_DMASR, DMA_SR_ERR_IRQ);
            return ACCEL_ERR_DMA;
        }
        if (sr & DMA_SR_IOC_IRQ) {
            REG_WRITE(DMA_BASE, DMA_S2MM_DMASR, DMA_SR_IOC_IRQ);
            return ACCEL_OK;
        }
    }
    return ACCEL_ERR_TIMEOUT;
}

/* Copy data to the DDR input buffer, flush cache, and send via MM2S.
 * Blocks until the MM2S transfer completes or times out.                    */
static accel_status_t dma_send_mm2s(const int8_t *data, uint32_t len)
{
    /* Copy input into the DMA source buffer in DDR */
    memcpy((void *)(uintptr_t)INPUT_BUF_ADDR, data, len);

    /* Flush cache so DMA reads current data from DDR */
    Xil_DCacheFlushRange(INPUT_BUF_ADDR, len);

    REG_WRITE(DMA_BASE, DMA_MM2S_SA, INPUT_BUF_ADDR);
    REG_WRITE(DMA_BASE, DMA_MM2S_DMACR, DMA_CR_RS);
    REG_WRITE(DMA_BASE, DMA_MM2S_LENGTH, len);   /* triggers transfer       */

    /* Poll for MM2S completion */
    for (uint32_t i = 0; i < DMA_XFER_TIMEOUT; i++) {
        uint32_t sr = REG_READ(DMA_BASE, DMA_MM2S_DMASR);
        if (sr & DMA_SR_ERR_IRQ) {
            REG_WRITE(DMA_BASE, DMA_MM2S_DMASR, DMA_SR_ERR_IRQ);
            return ACCEL_ERR_DMA;
        }
        if (sr & DMA_SR_IOC_IRQ) {
            REG_WRITE(DMA_BASE, DMA_MM2S_DMASR, DMA_SR_IOC_IRQ);
            return ACCEL_OK;
        }
    }
    return ACCEL_ERR_TIMEOUT;
}

/* =======================================================================
 * Core driver -- lifecycle
 * =======================================================================*/

accel_status_t accel_init(void)
{
    /* Reset both DMA channels */
    if (dma_reset_channel(DMA_MM2S_DMACR) != 0)
        return ACCEL_ERR_DMA;
    if (dma_reset_channel(DMA_S2MM_DMACR) != 0)
        return ACCEL_ERR_DMA;

    /* Reset accelerator */
    return accel_reset();
}

accel_status_t accel_reset(void)
{
    /* CTRL bit 1 = SOFT_RST (self-clearing in RTL) */
    REG_WRITE(ACCEL_BASE, ACCEL_CTRL, CTRL_SOFT_RST);

    /* Poll until STATUS shows IDLE (bit 0) */
    for (int i = 0; i < ACCEL_RESET_TIMEOUT; i++) {
        if (REG_READ(ACCEL_BASE, ACCEL_STATUS) & STATUS_IDLE)
            return ACCEL_OK;
    }
    return ACCEL_ERR_TIMEOUT;
}

bool accel_is_ready(void)
{
    uint32_t st = REG_READ(ACCEL_BASE, ACCEL_STATUS);
    return (st & STATUS_IDLE) && !(st & STATUS_BUSY);
}

/* =======================================================================
 * Blocking inference
 * =======================================================================*/

accel_status_t accel_run_inference(const int8_t *input, int8_t *output)
{
    accel_status_t rc;

    if (!accel_is_ready())
        return ACCEL_ERR_BUSY;

    /* Step 1: Arm S2MM so output path is ready */
    rc = dma_arm_s2mm(OUTPUT_SIZE_BYTES);
    if (rc != ACCEL_OK) return rc;

    /* Step 2: Start accelerator (CTRL bit 0, self-clearing) */
    REG_WRITE(ACCEL_BASE, ACCEL_CTRL, CTRL_START);

    /* Step 3: Send input via MM2S (blocking) */
    rc = dma_send_mm2s(input, INPUT_SIZE_BYTES);
    if (rc != ACCEL_OK) {
        accel_reset();
        return rc;
    }

    /* Step 4: Wait for accelerator DONE */
    for (uint32_t i = 0; i < ACCEL_DONE_TIMEOUT; i++) {
        uint32_t st = REG_READ(ACCEL_BASE, ACCEL_STATUS);
        if (st & STATUS_ERROR) {
            accel_reset();
            return ACCEL_ERR_HARDWARE;
        }
        if (st & STATUS_DONE)
            break;
        if (i == ACCEL_DONE_TIMEOUT - 1) {
            accel_reset();
            return ACCEL_ERR_TIMEOUT;
        }
    }

    /* Step 5: Wait for S2MM DMA to complete.
     * The accelerator asserts TLAST on the final output beat, which
     * signals S2MM to complete.  If S2MM never completes, TLAST is
     * missing from the accelerator -- that is a hardware bug.               */
    rc = dma_wait_s2mm();
    if (rc != ACCEL_OK) {
        accel_reset();
        return rc;
    }

    /* Step 6: Invalidate cache (DMA wrote behind it) and copy output */
    Xil_DCacheInvalidateRange(OUTPUT_BUF_ADDR, OUTPUT_SIZE_BYTES);
    memcpy(output, (void *)(uintptr_t)OUTPUT_BUF_ADDR, OUTPUT_SIZE_BYTES);

    return ACCEL_OK;
}

/* =======================================================================
 * Non-blocking (interrupt-driven) inference
 * =======================================================================*/

accel_status_t accel_start_inference(const int8_t *input, accel_done_cb_t cb)
{
    accel_status_t rc;

    if (!accel_is_ready())
        return ACCEL_ERR_BUSY;

    g_inference_done  = false;
    g_inference_error = false;
    g_done_callback   = cb;

    /* Arm S2MM with interrupt-on-complete */
    Xil_DCacheInvalidateRange(OUTPUT_BUF_ADDR, OUTPUT_SIZE_BYTES);
    REG_WRITE(DMA_BASE, DMA_S2MM_DA, OUTPUT_BUF_ADDR);
    REG_WRITE(DMA_BASE, DMA_S2MM_DMACR, DMA_CR_RS | DMA_CR_IOC_IRQ_EN);
    REG_WRITE(DMA_BASE, DMA_S2MM_LENGTH, OUTPUT_SIZE_BYTES);

    /* Enable accelerator DONE + ERROR interrupts */
    accel_enable_interrupts(true, true);

    /* Start accelerator (CTRL bit 0) */
    REG_WRITE(ACCEL_BASE, ACCEL_CTRL, CTRL_START);

    /* Send input via MM2S (blocks on the send, but accelerator processes
     * data as it arrives and output streams back via S2MM)                  */
    rc = dma_send_mm2s(input, INPUT_SIZE_BYTES);
    if (rc != ACCEL_OK) {
        accel_reset();
        return rc;
    }

    return ACCEL_OK;
}

accel_status_t accel_poll_completion(void)
{
    if (g_inference_error)
        return ACCEL_ERR_HARDWARE;
    if (g_inference_done)
        return ACCEL_OK;
    return ACCEL_ERR_BUSY;
}

accel_status_t accel_get_output(int8_t *output)
{
    if (g_inference_error)
        return ACCEL_ERR_HARDWARE;
    if (!g_inference_done)
        return ACCEL_ERR_BUSY;

    /* S2MM should already be done if accelerator DONE fired */
    accel_status_t rc = dma_wait_s2mm();
    if (rc != ACCEL_OK)
        return rc;

    Xil_DCacheInvalidateRange(OUTPUT_BUF_ADDR, OUTPUT_SIZE_BYTES);
    memcpy(output, (void *)(uintptr_t)OUTPUT_BUF_ADDR, OUTPUT_SIZE_BYTES);

    return ACCEL_OK;
}

/* =======================================================================
 * Interrupt handling
 * =======================================================================*/

void accel_enable_interrupts(bool done_irq, bool error_irq)
{
    uint32_t en = 0;
    if (done_irq)  en |= IRQ_DONE;
    if (error_irq) en |= IRQ_ERROR;
    REG_WRITE(ACCEL_BASE, ACCEL_IRQ_EN, en);
}

/*
 * ISR entry point.  Called from the GIC handler registered in interrupt.c.
 * Reads IRQ_STATUS, sets internal flags, clears pending interrupts (W1C),
 * and fires the user callback if registered.
 *
 * The 'data' parameter is the CallBackRef passed to XScuGic_Connect
 * (unused here, but required by the Xilinx ISR signature).
 */
void accel_isr(void *data)
{
    (void)data;

    uint32_t irq_st = REG_READ(ACCEL_BASE, ACCEL_IRQ_STATUS);

    if (irq_st & IRQ_DONE)
        g_inference_done = true;

    if (irq_st & IRQ_ERROR)
        g_inference_error = true;

    /* Write-1-to-clear all pending interrupt bits */
    REG_WRITE(ACCEL_BASE, ACCEL_IRQ_STATUS, irq_st);

    /* Fire callback if registered */
    if (g_done_callback) {
        accel_status_t cb_st = g_inference_error ? ACCEL_ERR_HARDWARE
                                                 : ACCEL_OK;
        g_done_callback(cb_st);
    }
}

/* =======================================================================
 * Diagnostics
 * =======================================================================*/

uint32_t accel_get_cycle_count(void)
{
    return REG_READ(ACCEL_BASE, ACCEL_CYCLE_COUNT);
}

uint32_t accel_get_inference_count(void)
{
    return REG_READ(ACCEL_BASE, ACCEL_INF_COUNT);
}

uint32_t accel_get_version(void)
{
    return REG_READ(ACCEL_BASE, ACCEL_VERSION);
}

uint32_t accel_get_error_code(void)
{
    return REG_READ(ACCEL_BASE, ACCEL_ERROR_CODE);
}

void accel_dump_regs(void)
{
    xil_printf("---- Accelerator CSR (base 0x%08x) ----\r\n", ACCEL_BASE);
    xil_printf("  CTRL:         0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_CTRL));
    xil_printf("  STATUS:       0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_STATUS));
    xil_printf("  IRQ_EN:       0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_IRQ_EN));
    xil_printf("  IRQ_STATUS:   0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_IRQ_STATUS));
    xil_printf("  CYCLE_COUNT:  %u\r\n",     REG_READ(ACCEL_BASE, ACCEL_CYCLE_COUNT));
    xil_printf("  INF_COUNT:    %u\r\n",     REG_READ(ACCEL_BASE, ACCEL_INF_COUNT));
    xil_printf("  VERSION:      0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_VERSION));
    xil_printf("  SCRATCH:      0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_SCRATCH));
    xil_printf("  ERROR_CODE:   0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_ERROR_CODE));
    xil_printf("  LAYER_STATUS: 0x%08x\r\n", REG_READ(ACCEL_BASE, ACCEL_LAYER_STATUS));
}

void dma_dump_status(void)
{
    uint32_t mm2s_sr = REG_READ(DMA_BASE, DMA_MM2S_DMASR);
    uint32_t s2mm_sr = REG_READ(DMA_BASE, DMA_S2MM_DMASR);

    xil_printf("---- DMA Status (base 0x%08x) ----\r\n", DMA_BASE);
    xil_printf("  MM2S SR: 0x%08x [Halted=%d Idle=%d IOC=%d Err=%d]\r\n",
               mm2s_sr,
               !!(mm2s_sr & DMA_SR_HALTED),
               !!(mm2s_sr & DMA_SR_IDLE),
               !!(mm2s_sr & DMA_SR_IOC_IRQ),
               !!(mm2s_sr & DMA_SR_ERR_IRQ));

    xil_printf("  S2MM SR: 0x%08x [Halted=%d Idle=%d IOC=%d Err=%d]\r\n",
               s2mm_sr,
               !!(s2mm_sr & DMA_SR_HALTED),
               !!(s2mm_sr & DMA_SR_IDLE),
               !!(s2mm_sr & DMA_SR_IOC_IRQ),
               !!(s2mm_sr & DMA_SR_ERR_IRQ));
}
