; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 4
; This file is licensed under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
;
; RUN: llc -mtriple=aie2 -O2 --issue-limit=1 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; Test to check for liveness of simplifiable reserved regs (i.e. crsat and crrnd in this test)
; across call boundaries.

define void @caller1() {
; CHECK-LABEL: caller1:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; jl #callee1; nops
; CHECK-NEXT:    nop // Delay Slot 5
; CHECK-NEXT:    paddb [sp], #32 // Delay Slot 4
; CHECK-NEXT:    st lr, [sp, #-32] // 4-byte Folded Spill Delay Slot 3
; CHECK-NEXT:    mov crSat, #1 // Delay Slot 2
; CHECK-NEXT:    mov crRnd, #12 // Delay Slot 1
; CHECK-NEXT:    lda lr, [sp, #-32] // 4-byte Folded Reload
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK-NEXT:    ret lr
; CHECK-NEXT:    nop // Delay Slot 5
; CHECK-NEXT:    nop // Delay Slot 4
; CHECK-NEXT:    nop // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    paddb [sp], #-32 // Delay Slot 1
entry:
	tail call void @llvm.aie2.set.ctrl.reg(i32 9, i32 1)
  	tail call void @llvm.aie2.set.ctrl.reg(i32 6, i32 12)
  	tail call void @callee1()
  	ret void
}

define void @callee1() {
; CHECK-LABEL: callee1:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    mova r0, #1; nopb ; nopxm ; nops
; CHECK-NEXT:    ret lr
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    vsrs.d8.s32 wh0, cm0, s0 // Delay Slot 4
; CHECK-NEXT:    nop // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    nop // Delay Slot 1
entry:
    %0 = tail call noundef <16 x i64> @llvm.aie2.v32acc32()
	%1 = tail call noundef <32 x i8> @llvm.aie2.I256.v32.acc32.srs(<16 x i64> %0, i32 1, i32 0)
	ret void
}

declare <32 x i8> @llvm.aie2.I256.v32.acc32.srs(<16 x i64>, i32, i32)

declare <16 x i64> @llvm.aie2.v32acc32()

declare void @llvm.aie2.set.ctrl.reg(i32, i32)
