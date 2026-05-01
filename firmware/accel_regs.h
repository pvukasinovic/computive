/*
 * accel_regs.h -- Memory-mapped register definitions for the ML accelerator IP
 *
 * Target: Kria KV260 (Zynq UltraScale+ MPSoC, ARM Cortex-A53)
 * Board part: xilinx.com:kv260_som:part0:1.4
 *
 * Register offsets and bit fields are derived from the RTL source:
 *   rtl/interface/axi_lite_ctrl.sv (lines 59-68 for addresses,
 *   lines 154-156 for CTRL bits, line 172 for STATUS bits).
 *
 * Address map is from the Vivado block design address editor.
 * DMA register offsets follow the Xilinx AXI DMA IP (PG021) standard.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ACCEL_REGS_H
#define ACCEL_REGS_H

#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Base addresses (must match Vivado address editor)
 * ---------------------------------------------------------------------------*/
#define ACCEL_BASE       0xA0000000UL   /* Accelerator CSR, 4 KB             */
#define DMA_BASE         0xA0010000UL   /* AXI DMA controller, 4 KB          */
#define INPUT_BUF_ADDR   0x00100000UL   /* DDR input buffer, 64-byte aligned */
#define OUTPUT_BUF_ADDR  0x00200000UL   /* DDR output buffer, 64-byte aligned*/

/* ---------------------------------------------------------------------------
 * Accelerator register offsets (from axi_lite_ctrl.sv:59-68)
 * ---------------------------------------------------------------------------*/
#define ACCEL_CTRL          0x00
#define ACCEL_STATUS        0x04
#define ACCEL_IRQ_EN        0x08
#define ACCEL_IRQ_STATUS    0x0C
#define ACCEL_CYCLE_COUNT   0x10
#define ACCEL_INF_COUNT     0x14
#define ACCEL_VERSION       0x18
#define ACCEL_SCRATCH       0x1C
#define ACCEL_ERROR_CODE    0x20
#define ACCEL_LAYER_STATUS  0x24

/* ---------------------------------------------------------------------------
 * CTRL register bits -- matches axi_lite_ctrl.sv:154-156
 *   assign ctrl_start      = reg_ctrl[0];   // self-clearing
 *   assign ctrl_soft_rst   = reg_ctrl[1];   // self-clearing
 *   assign ctrl_continuous = reg_ctrl[2];   // persistent
 * NOTE: There is NO ABORT bit in the RTL.
 * ---------------------------------------------------------------------------*/
#define CTRL_START       (1U << 0)
#define CTRL_SOFT_RST    (1U << 1)
#define CTRL_CONTINUOUS  (1U << 2)

/* ---------------------------------------------------------------------------
 * STATUS register bits -- matches axi_lite_ctrl.sv:172
 *   {28'b0, status_error, status_done, status_busy, status_idle}
 * ---------------------------------------------------------------------------*/
#define STATUS_IDLE   (1U << 0)
#define STATUS_BUSY   (1U << 1)
#define STATUS_DONE   (1U << 2)
#define STATUS_ERROR  (1U << 3)

/* ---------------------------------------------------------------------------
 * IRQ_EN / IRQ_STATUS bits
 * IRQ_STATUS is W1C (write-1-to-clear), see axi_lite_ctrl.sv:145.
 * ---------------------------------------------------------------------------*/
#define IRQ_DONE   (1U << 0)
#define IRQ_ERROR  (1U << 1)
#define IRQ_ALL    (IRQ_DONE | IRQ_ERROR)

/* ---------------------------------------------------------------------------
 * Error codes (read from ERROR_CODE register when STATUS_ERROR is set)
 * ---------------------------------------------------------------------------*/
#define ERR_NONE            0x00
#define ERR_INPUT_TIMEOUT   0x01
#define ERR_OUTPUT_OVERFLOW 0x02
#define ERR_INTERNAL        0x03

/* ---------------------------------------------------------------------------
 * AXI DMA register offsets (Xilinx PG021)
 * ---------------------------------------------------------------------------*/
#define DMA_MM2S_DMACR   0x00   /* MM2S control                            */
#define DMA_MM2S_DMASR   0x04   /* MM2S status                             */
#define DMA_MM2S_SA      0x18   /* MM2S source address                     */
#define DMA_MM2S_LENGTH  0x28   /* MM2S transfer length (triggers xfer)    */
#define DMA_S2MM_DMACR   0x30   /* S2MM control                            */
#define DMA_S2MM_DMASR   0x34   /* S2MM status                             */
#define DMA_S2MM_DA      0x48   /* S2MM destination address                */
#define DMA_S2MM_LENGTH  0x58   /* S2MM transfer length (triggers xfer)    */

/* DMA control bits */
#define DMA_CR_RS          (1U << 0)   /* Run/Stop                         */
#define DMA_CR_RESET       (1U << 2)   /* Soft reset (self-clearing)       */
#define DMA_CR_IOC_IRQ_EN  (1U << 12)  /* Interrupt on complete enable     */
#define DMA_CR_ERR_IRQ_EN  (1U << 14)  /* Interrupt on error enable        */

/* DMA status bits */
#define DMA_SR_HALTED      (1U << 0)   /* Channel halted                   */
#define DMA_SR_IDLE        (1U << 1)   /* Channel idle                     */
#define DMA_SR_IOC_IRQ     (1U << 12)  /* Transfer complete (W1C)          */
#define DMA_SR_ERR_IRQ     (1U << 14)  /* Error occurred (W1C)             */

/* ---------------------------------------------------------------------------
 * Data sizes for the AD model (640-element INT8 input and output)
 * ---------------------------------------------------------------------------*/
#define INPUT_SIZE_BYTES    640
#define OUTPUT_SIZE_BYTES   640

/* Cache line size for Cortex-A53 (UltraScale+) */
#define CACHE_LINE_BYTES    64

/* ---------------------------------------------------------------------------
 * Register access macros
 * Uses direct physical address access, valid on bare-metal standalone BSP
 * with 1:1 MMU mapping. For Linux, use mmap'd virtual addresses instead.
 * ---------------------------------------------------------------------------*/
#define REG_WRITE(base, offset, val) \
    (*(volatile uint32_t *)((uintptr_t)(base) + (offset)) = (val))

#define REG_READ(base, offset) \
    (*(volatile uint32_t *)((uintptr_t)(base) + (offset)))

#endif /* ACCEL_REGS_H */
