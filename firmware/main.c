/*
 * main.c -- Standalone self-test and measurement harness for the ML accelerator
 *
 * Target: Kria KV260 (Zynq UltraScale+ MPSoC, ARM Cortex-A53)
 * Board part: xilinx.com:kv260_som:part0:1.4
 *
 * Boot sequence:
 *   1. FSBL initializes PS (clocks, DDR, MIO), loads bitstream, jumps here
 *   2. init_platform() enables caches and UART
 *   3. setup_interrupts() configures GIC for PL IRQs
 *   4. accel_init() resets DMA channels and accelerator
 *   5. Scratch register connectivity test
 *   6. Print IP version
 *   7. Run N=100 inferences with test vector
 *   8. Compare output vs golden reference, report PASS/FAIL
 *   9. Throughput measurement via XTime (750 MHz timer on A53)
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform.h"
#include "xil_printf.h"
#include "xtime_l.h"
#include "accel.h"
#include "accel_regs.h"
#include "test_vectors.h"
#include <string.h>

/* Number of inferences for the correctness + throughput loop */
#define NUM_INFERENCES     100

/* Warm-up inferences before measurement (fills caches, DMA pipelines) */
#define WARMUP_INFERENCES  5

/* XTime counter runs at 750 MHz on UltraScale+ (CPU_FREQ / 2 = 1.5G / 2) */
#define TIMER_FREQ_MHZ     750

/* Forward declaration (defined in interrupt.c) */
extern int setup_interrupts(void);

/* -----------------------------------------------------------------------
 * Helper: compare output against golden reference
 * Returns the number of mismatched bytes.
 * -----------------------------------------------------------------------*/
static int verify_output(const int8_t *output, const int8_t *expected,
                         int len, int max_print)
{
    int errors = 0;
    for (int i = 0; i < len; i++) {
        if (output[i] != expected[i]) {
            errors++;
            if (errors <= max_print) {
                xil_printf("  Mismatch [%d]: got %d, expected %d\r\n",
                           i, (int)output[i], (int)expected[i]);
            }
        }
    }
    return errors;
}

/* -----------------------------------------------------------------------
 * main
 * -----------------------------------------------------------------------*/
int main(void)
{
    accel_status_t rc;

    /* ---- Platform init ------------------------------------------------- */
    init_platform();
    xil_printf("\r\n");
    xil_printf("========================================\r\n");
    xil_printf("  MLASIC Accelerator Self-Test\r\n");
    xil_printf("  Target: Kria KV260 (Cortex-A53)\r\n");
    xil_printf("========================================\r\n");

    /* ---- Interrupt setup ----------------------------------------------- */
    if (setup_interrupts() != 0) {
        xil_printf("FATAL: Interrupt setup failed\r\n");
        goto fail;
    }
    xil_printf("[OK] Interrupt controller initialized\r\n");

    /* ---- Accelerator + DMA init ---------------------------------------- */
    rc = accel_init();
    if (rc != ACCEL_OK) {
        xil_printf("FATAL: accel_init failed (rc=%d)\r\n", (int)rc);
        goto fail;
    }
    xil_printf("[OK] Accelerator and DMA initialized\r\n");

    /* ---- Scratch register connectivity test ---------------------------- */
    {
        const uint32_t pattern = 0xDEADBEEF;
        REG_WRITE(ACCEL_BASE, ACCEL_SCRATCH, pattern);
        uint32_t readback = REG_READ(ACCEL_BASE, ACCEL_SCRATCH);
        if (readback != pattern) {
            xil_printf("FATAL: Scratch register test failed\r\n");
            xil_printf("  Wrote 0x%08x, read 0x%08x\r\n", pattern, readback);
            xil_printf("  Check bitstream and AXI-Lite connection\r\n");
            goto fail;
        }
        /* Write a second pattern to exercise more bits */
        REG_WRITE(ACCEL_BASE, ACCEL_SCRATCH, 0x12345678);
        readback = REG_READ(ACCEL_BASE, ACCEL_SCRATCH);
        if (readback != 0x12345678) {
            xil_printf("FATAL: Scratch register second pattern failed\r\n");
            xil_printf("  Wrote 0x12345678, read 0x%08x\r\n", readback);
            goto fail;
        }
        /* Clear scratch register */
        REG_WRITE(ACCEL_BASE, ACCEL_SCRATCH, 0x00000000);
    }
    xil_printf("[OK] Scratch register connectivity verified\r\n");

    /* ---- IP version ---------------------------------------------------- */
    {
        uint32_t ver = accel_get_version();
        xil_printf("[INFO] IP version: %u.%u.%u (raw 0x%08x)\r\n",
                   (ver >> 16) & 0xFF, (ver >> 8) & 0xFF, ver & 0xFF, ver);
    }

    /* ---- STATUS sanity check ------------------------------------------- */
    {
        uint32_t st = REG_READ(ACCEL_BASE, ACCEL_STATUS);
        xil_printf("[INFO] STATUS after reset: 0x%08x", st);
        if (st & STATUS_IDLE)  xil_printf(" IDLE");
        if (st & STATUS_BUSY)  xil_printf(" BUSY");
        if (st & STATUS_DONE)  xil_printf(" DONE");
        if (st & STATUS_ERROR) xil_printf(" ERROR");
        xil_printf("\r\n");
        if (!(st & STATUS_IDLE)) {
            xil_printf("WARNING: Accelerator not IDLE after reset\r\n");
        }
    }

    /* ---- Warm-up inferences -------------------------------------------- */
    xil_printf("[INFO] Running %d warm-up inferences...\r\n", WARMUP_INFERENCES);
    {
        int8_t discard[OUTPUT_SIZE_BYTES];
        for (int i = 0; i < WARMUP_INFERENCES; i++) {
            rc = accel_run_inference(test_input, discard);
            if (rc != ACCEL_OK) {
                xil_printf("WARNING: Warm-up inference %d failed (rc=%d)\r\n",
                           i, (int)rc);
                accel_dump_regs();
                dma_dump_status();
                break;
            }
        }
    }

    /* ---- Correctness test: N inferences -------------------------------- */
    xil_printf("[INFO] Running %d inferences for correctness test...\r\n",
               NUM_INFERENCES);
    {
        int8_t output[OUTPUT_SIZE_BYTES];
        int pass_count = 0;
        int fail_count = 0;

        XTime t_start, t_end;
        XTime_GetTime(&t_start);

        for (int i = 0; i < NUM_INFERENCES; i++) {
            rc = accel_run_inference(test_input, output);
            if (rc != ACCEL_OK) {
                xil_printf("  Inference %d error (rc=%d, hw_err=0x%02x)\r\n",
                           i, (int)rc, accel_get_error_code());
                fail_count++;
                /* Attempt recovery and continue */
                accel_reset();
                continue;
            }

            int errors = verify_output(output, expected_output,
                                       OUTPUT_SIZE_BYTES, 5);
            if (errors == 0) {
                pass_count++;
            } else {
                fail_count++;
                if (fail_count <= 3) {
                    xil_printf("  Inference %d: %d/%d mismatches\r\n",
                               i, errors, OUTPUT_SIZE_BYTES);
                }
            }
        }

        XTime_GetTime(&t_end);

        /* ---- Results --------------------------------------------------- */
        xil_printf("\r\n");
        xil_printf("---- Results ----\r\n");
        if (fail_count == 0) {
            xil_printf("PASS: %d/%d inferences matched golden reference\r\n",
                       pass_count, NUM_INFERENCES);
        } else {
            xil_printf("FAIL: %d passed, %d failed out of %d\r\n",
                       pass_count, fail_count, NUM_INFERENCES);
        }

        /* ---- Hardware cycle counter ------------------------------------ */
        uint32_t hw_cycles = accel_get_cycle_count();
        /* At 100 MHz PL clock: 1 cycle = 10 ns, so latency_us = cycles/100 */
        uint32_t hw_latency_us = hw_cycles / 100;
        xil_printf("HW cycle count (last inference): %u (%u us at 100 MHz)\r\n",
                   hw_cycles, hw_latency_us);
        xil_printf("Inference count (HW register):   %u\r\n",
                   accel_get_inference_count());

        /* ---- Wall-clock throughput (XTime, 750 MHz timer) -------------- */
        uint64_t elapsed = t_end - t_start;
        /* elapsed is in 750 MHz counts.  Total microseconds: */
        uint32_t total_us = (uint32_t)(elapsed / TIMER_FREQ_MHZ);
        /* Average latency per inference */
        uint32_t avg_us = (NUM_INFERENCES > 0) ? total_us / NUM_INFERENCES : 0;
        /* Throughput: inferences per second */
        uint32_t throughput = 0;
        if (total_us > 0)
            throughput = (uint32_t)((uint64_t)NUM_INFERENCES * 1000000ULL
                                   / total_us);

        xil_printf("\r\n---- Throughput ----\r\n");
        xil_printf("Total time for %d inferences: %u us\r\n",
                   NUM_INFERENCES, total_us);
        xil_printf("Average latency: %u us/inference\r\n", avg_us);
        xil_printf("Throughput: %u inferences/sec\r\n", throughput);
        xil_printf("  (includes DMA setup + cache management overhead)\r\n");
    }

    /* ---- Done ---------------------------------------------------------- */
    xil_printf("\r\n========================================\r\n");
    xil_printf("  Self-test complete\r\n");
    xil_printf("========================================\r\n");

    cleanup_platform();
    return 0;

fail:
    xil_printf("\r\nSelf-test ABORTED due to fatal error.\r\n");
    accel_dump_regs();
    dma_dump_status();
    cleanup_platform();
    return -1;
}
