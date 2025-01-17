; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
;
; This file is licensed under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
; RUN: llc < %s -verify-machineinstrs -mtriple=aie2 | FileCheck %s


define dso_local <8 x i64> @_Z9test_lupsDv8_ji(<8 x i32> noundef %a, i32 noundef %shft) local_unnamed_addr #0 {
; CHECK-LABEL: _Z9test_lupsDv8_ji:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopb ; nopa ; nops ; ret lr ; nopm ; nopv
; CHECK-NEXT:    nopx // Delay Slot 5
; CHECK-NEXT:    mov s0, r0 // Delay Slot 4
; CHECK-NEXT:    vups.s64.d32 bml0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    nop // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.acc64.v8.I256.ups(<8 x i32> %a, i32 %shft, i32 0)
  ret <8 x i64> %0
}

define dso_local <8 x i64> @_Z9test_lupsDv8_ji_1(<8 x i32> noundef %a, i32 noundef %shft) local_unnamed_addr #0 {
; CHECK-LABEL: _Z9test_lupsDv8_ji_1:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopb ; nopa ; nops ; ret lr ; nopm ; nopv
; CHECK-NEXT:    nopx // Delay Slot 5
; CHECK-NEXT:    mov s0, r0 // Delay Slot 4
; CHECK-NEXT:    vups.s64.s32 bml0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    nop // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.acc64.v8.I256.ups(<8 x i32> %a, i32 %shft, i32 1)
  ret <8 x i64> %0
}

define dso_local <8 x i64> @_Z9test_lupsDv8_iii(<8 x i32> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #0 {
; CHECK-LABEL: _Z9test_lupsDv8_iii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s64.d32 bml0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.acc64.v8.I256.ups(<8 x i32> %a, i32 %shft, i32 %sign)
  ret <8 x i64> %0
}

define dso_local <16 x i64> @_Z9test_supsDv32_hii(<32 x i8> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #1 {
; CHECK-LABEL: _Z9test_supsDv32_hii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s32.d8 cm0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <16 x i64> @llvm.aie2.acc32.v32.I256.ups(<32 x i8> %a, i32 %shft, i32 %sign)
  ret <16 x i64> %0
}

define dso_local <16 x i64> @_Z9test_lupsDv16_sii(<16 x i16> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #1 {
; CHECK-LABEL: _Z9test_lupsDv16_sii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s64.d16 cm0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <16 x i64> @llvm.aie2.acc64.v16.I256.ups(<16 x i16> %a, i32 %shft, i32 %sign)
  ret <16 x i64> %0
}

define dso_local <8 x i64> @_Z19test_ups_to_v8acc64Dv8_iii(<8 x i32> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #0 {
; CHECK-LABEL: _Z19test_ups_to_v8acc64Dv8_iii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s64.d32 bml0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.acc64.v8.I256.ups(<8 x i32> %a, i32 %shft, i32 %sign)
  ret <8 x i64> %0
}

define dso_local <8 x i64> @_Z20test_ups_to_v16acc32Dv16_tii(<16 x i16> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #0 {
; CHECK-LABEL: _Z20test_ups_to_v16acc32Dv16_tii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s32.d16 bml0, wl0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.acc32.v16.I256.ups(<16 x i16> %a, i32 %shft, i32 %sign)
  ret <8 x i64> %0
}

define dso_local <16 x i64> @_Z20test_ups_to_v16acc64Dv16_iii(<16 x i32> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #1 {
; CHECK-LABEL: _Z20test_ups_to_v16acc64Dv16_iii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s64.d32 cm0, x0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <16 x i64> @llvm.aie2.acc64.v16.I512.ups(<16 x i32> %a, i32 %shft, i32 %sign)
  ret <16 x i64> %0
}

define dso_local <16 x i64> @_Z20test_ups_to_v32acc32Dv32_sii(<32 x i16> noundef %a, i32 noundef %shft, i32 noundef %sign) local_unnamed_addr #1 {
; CHECK-LABEL: _Z20test_ups_to_v32acc32Dv32_sii:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopa ; nopb ; ret lr ; nopm ; nops
; CHECK-NEXT:    mov s0, r0 // Delay Slot 5
; CHECK-NEXT:    mov crUPSSign, r1 // Delay Slot 4
; CHECK-NEXT:    vups.s32.d16 cm0, x0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    mov crUPSSign, #0 // Delay Slot 1
entry:
  %0 = tail call <16 x i64> @llvm.aie2.acc32.v32.I512.ups(<32 x i16> %a, i32 %shft, i32 %sign)
  ret <16 x i64> %0
}

define <8 x i64> @test_ups_v16accfloat(<16 x bfloat> %a) {
; CHECK-LABEL: test_ups_v16accfloat:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    ret lr
; CHECK-NEXT:    nop // Delay Slot 5
; CHECK-NEXT:    nop // Delay Slot 4
; CHECK-NEXT:    nop // Delay Slot 3
; CHECK-NEXT:    vconv.fp32.bf16 bml0, wl0 // Delay Slot 2
; CHECK-NEXT:    nop // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.v16bf16.to.v16accfloat(<16 x bfloat> %a)
  ret <8 x i64> %0
}

define <8 x i64> @test_ups(<16 x bfloat> %a) {
; CHECK-LABEL: test_ups:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    ret lr
; CHECK-NEXT:    nop // Delay Slot 5
; CHECK-NEXT:    nop // Delay Slot 4
; CHECK-NEXT:    nop // Delay Slot 3
; CHECK-NEXT:    vconv.fp32.bf16 bml0, wl0 // Delay Slot 2
; CHECK-NEXT:    nop // Delay Slot 1
entry:
  %0 = tail call <8 x i64> @llvm.aie2.v16bf16.to.v16accfloat(<16 x bfloat> %a)
  ret <8 x i64> %0
}

define <16 x i64> @test_ups_to_v32acc32(<32 x i16> %a, i32 %shft) {
; CHECK-LABEL: test_ups_to_v32acc32:
; CHECK:         .p2align 4
; CHECK-NEXT:  // %bb.0: // %entry
; CHECK-NEXT:    nopb ; nopa ; nops ; ret lr ; nopm ; nopv
; CHECK-NEXT:    nopx // Delay Slot 5
; CHECK-NEXT:    mov s0, r0 // Delay Slot 4
; CHECK-NEXT:    vups.s32.d16 cm0, x0, s0 // Delay Slot 3
; CHECK-NEXT:    nop // Delay Slot 2
; CHECK-NEXT:    nop // Delay Slot 1
entry:
  %0 = tail call <16 x i64> @llvm.aie2.acc32.v32.I512.ups(<32 x i16> %a, i32 %shft, i32 0)
  ret <16 x i64> %0
}

declare <8 x i64> @llvm.aie2.acc64.v8.I256.ups(<8 x i32>, i32, i32) #2
declare <16 x i64> @llvm.aie2.acc32.v32.I256.ups(<32 x i8>, i32, i32) #2
declare <16 x i64> @llvm.aie2.acc64.v16.I256.ups(<16 x i16>, i32, i32) #2
declare <8 x i64> @llvm.aie2.acc32.v16.I256.ups(<16 x i16>, i32, i32) #2
declare <16 x i64> @llvm.aie2.acc64.v16.I512.ups(<16 x i32>, i32, i32) #2
declare <16 x i64> @llvm.aie2.acc32.v32.I512.ups(<32 x i16>, i32, i32) #2
declare <8 x i64> @llvm.aie2.v16bf16.to.v16accfloat(<16 x bfloat>)
