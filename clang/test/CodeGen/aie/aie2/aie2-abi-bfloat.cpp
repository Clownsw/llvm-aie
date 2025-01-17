// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
//===- aie2-abi-bfloat.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: %clang --target=aie2 -nostdlibinc  -D_LIBCPP_HAS_THREAD_API_PTHREAD -S -emit-llvm %s -o - | FileCheck %s

#include <stdint.h>

extern "C" {

/****** bfloat float vector ******/

// CHECK-LABEL: @ret_v8bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RET:%.*]] = alloca <8 x bfloat>, align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load volatile <8 x bfloat>, ptr [[RET]], align 16
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
v8bfloat16 ret_v8bfloat16(void) {
  volatile v8bfloat16 ret;
  return ret;
}
// CHECK-LABEL: @pass_v8bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DOTADDR:%.*]] = alloca <8 x bfloat>, align 16
// CHECK-NEXT:    store <8 x bfloat> [[TMP0:%.*]], ptr [[DOTADDR]], align 16
// CHECK-NEXT:    ret void
//
void pass_v8bfloat16(v8bfloat16) {}

// CHECK-LABEL: @ret_v16bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RET:%.*]] = alloca <16 x bfloat>, align 32
// CHECK-NEXT:    [[TMP0:%.*]] = load volatile <16 x bfloat>, ptr [[RET]], align 32
// CHECK-NEXT:    ret <16 x bfloat> [[TMP0]]
//
v16bfloat16 ret_v16bfloat16(void) {
  volatile v16bfloat16 ret;
  return ret;
}
// CHECK-LABEL: @pass_v16bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DOTADDR:%.*]] = alloca <16 x bfloat>, align 32
// CHECK-NEXT:    store <16 x bfloat> [[TMP0:%.*]], ptr [[DOTADDR]], align 32
// CHECK-NEXT:    ret void
//
void pass_v16bfloat16(v16bfloat16) {}

// CHECK-LABEL: @ret_v32bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RET:%.*]] = alloca <32 x bfloat>, align 32
// CHECK-NEXT:    [[TMP0:%.*]] = load volatile <32 x bfloat>, ptr [[RET]], align 32
// CHECK-NEXT:    ret <32 x bfloat> [[TMP0]]
//
v32bfloat16 ret_v32bfloat16(void) {
  volatile v32bfloat16 ret;
  return ret;
}
// CHECK-LABEL: @pass_v32bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DOTADDR:%.*]] = alloca <32 x bfloat>, align 32
// CHECK-NEXT:    store <32 x bfloat> [[TMP0:%.*]], ptr [[DOTADDR]], align 32
// CHECK-NEXT:    ret void
//
void pass_v32bfloat16(v32bfloat16) {}

// CHECK-LABEL: @ret_v64bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RET:%.*]] = alloca <64 x bfloat>, align 32
// CHECK-NEXT:    [[TMP0:%.*]] = load volatile <64 x bfloat>, ptr [[RET]], align 32
// CHECK-NEXT:    ret <64 x bfloat> [[TMP0]]
//
v64bfloat16 ret_v64bfloat16(void) {
  volatile v64bfloat16 ret;
  return ret;
}
// CHECK-LABEL: @pass_v64bfloat16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DOTADDR:%.*]] = alloca <64 x bfloat>, align 32
// CHECK-NEXT:    store <64 x bfloat> [[TMP0:%.*]], ptr [[DOTADDR]], align 32
// CHECK-NEXT:    ret void
//
void pass_v64bfloat16(v64bfloat16) {}

}
