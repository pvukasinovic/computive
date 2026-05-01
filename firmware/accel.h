/*
 * accel.h -- Driver API for the ML accelerator IP
 *
 * Target: Kria KV260 (Zynq UltraScale+ MPSoC, ARM Cortex-A53)
 *
 * Provides blocking and interrupt-driven inference paths, DMA management,
 * and diagnostic register access.  All functions return accel_status_t to
 * indicate success or the class of error encountered.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ACCEL_H
#define ACCEL_H

#include <stdint.h>
#include <stdbool.h>

/* ---------------------------------------------------------------------------
 * Return codes
 * ---------------------------------------------------------------------------*/
typedef enum {
    ACCEL_OK = 0,
    ACCEL_ERR_BUSY,
    ACCEL_ERR_TIMEOUT,
    ACCEL_ERR_DMA,
    ACCEL_ERR_HARDWARE
} accel_status_t;

/* Callback type for interrupt-driven inference completion.
 * Invoked from ISR context -- keep the handler short.                       */
typedef void (*accel_done_cb_t)(accel_status_t status);

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------------*/

/* Initialize accelerator and DMA to a known-good state.
 * Must be called once before any other API function.                        */
accel_status_t accel_init(void);

/* Soft-reset accelerator via CTRL_SOFT_RST (bit 1, self-clearing).
 * Also resets both DMA channels.  Call after any error to recover.          */
accel_status_t accel_reset(void);

/* Return true if accelerator STATUS shows IDLE and not BUSY.                */
bool accel_is_ready(void);

/* ---------------------------------------------------------------------------
 * Blocking inference
 * ---------------------------------------------------------------------------*/

/* Run a single inference (blocking / polled).
 *   input  -- pointer to INPUT_SIZE_BYTES of INT8 values
 *   output -- pointer to OUTPUT_SIZE_BYTES of INT8 values (written on success)
 * Sequence: arm S2MM -> START -> MM2S send -> poll DONE -> wait S2MM ->
 *           invalidate cache -> copy output.
 * On error, the driver resets the accelerator before returning.             */
accel_status_t accel_run_inference(const int8_t *input, int8_t *output);

/* ---------------------------------------------------------------------------
 * Non-blocking (interrupt-driven) inference
 * ---------------------------------------------------------------------------*/

/* Start an inference without blocking.
 *   input -- pointer to INPUT_SIZE_BYTES (must remain valid until done)
 *   cb    -- completion callback (ISR context), or NULL to poll
 * Returns ACCEL_OK if inference was successfully started.                   */
accel_status_t accel_start_inference(const int8_t *input, accel_done_cb_t cb);

/* Poll for completion of a non-blocking inference.
 * Returns ACCEL_OK if done, ACCEL_ERR_BUSY if still running.               */
accel_status_t accel_poll_completion(void);

/* Copy output data after non-blocking inference completes.
 * Only valid after accel_poll_completion() returns ACCEL_OK or after the
 * completion callback has fired.                                            */
accel_status_t accel_get_output(int8_t *output);

/* ---------------------------------------------------------------------------
 * Interrupt control
 * ---------------------------------------------------------------------------*/

/* Enable or disable accelerator DONE and ERROR interrupts in IRQ_EN.        */
void accel_enable_interrupts(bool done_irq, bool error_irq);

/* ISR entry point -- read IRQ_STATUS, set flags, W1C, fire callback.
 * Must be registered with the GIC via setup_interrupts().                   */
void accel_isr(void *data);

/* ---------------------------------------------------------------------------
 * Diagnostics
 * ---------------------------------------------------------------------------*/
uint32_t accel_get_cycle_count(void);
uint32_t accel_get_inference_count(void);
uint32_t accel_get_version(void);
uint32_t accel_get_error_code(void);

/* Dump all accelerator CSR registers over UART (xil_printf).                */
void accel_dump_regs(void);

/* Dump MM2S and S2MM DMA status registers over UART.                        */
void dma_dump_status(void);

#endif /* ACCEL_H */
