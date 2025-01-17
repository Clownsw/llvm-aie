; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 3
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s

define void @main(i1 %arg) #0 {
; CHECK-LABEL: main:
; CHECK:       ; %bb.0: ; %bb
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_xor_saveexec_b64 s[4:5], -1
; CHECK-NEXT:    buffer_store_dword v8, off, s[0:3], s32 ; 4-byte Folded Spill
; CHECK-NEXT:    buffer_store_dword v4, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; CHECK-NEXT:    buffer_store_dword v3, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; CHECK-NEXT:    s_mov_b64 exec, s[4:5]
; CHECK-NEXT:    v_writelane_b32 v8, s30, 0
; CHECK-NEXT:    v_writelane_b32 v8, s31, 1
; CHECK-NEXT:    v_writelane_b32 v8, s36, 2
; CHECK-NEXT:    v_writelane_b32 v8, s37, 3
; CHECK-NEXT:    v_writelane_b32 v8, s38, 4
; CHECK-NEXT:    v_writelane_b32 v8, s39, 5
; CHECK-NEXT:    v_writelane_b32 v8, s40, 6
; CHECK-NEXT:    v_writelane_b32 v8, s41, 7
; CHECK-NEXT:    v_writelane_b32 v8, s42, 8
; CHECK-NEXT:    v_writelane_b32 v8, s43, 9
; CHECK-NEXT:    v_writelane_b32 v8, s44, 10
; CHECK-NEXT:    v_writelane_b32 v8, s45, 11
; CHECK-NEXT:    v_writelane_b32 v8, s46, 12
; CHECK-NEXT:    v_writelane_b32 v8, s47, 13
; CHECK-NEXT:    v_writelane_b32 v8, s48, 14
; CHECK-NEXT:    v_writelane_b32 v8, s49, 15
; CHECK-NEXT:    s_getpc_b64 s[24:25]
; CHECK-NEXT:    v_writelane_b32 v8, s50, 16
; CHECK-NEXT:    s_movk_i32 s4, 0xf0
; CHECK-NEXT:    s_mov_b32 s5, s24
; CHECK-NEXT:    v_writelane_b32 v8, s51, 17
; CHECK-NEXT:    s_load_dwordx16 s[36:51], s[4:5], 0x0
; CHECK-NEXT:    ; implicit-def: $vgpr4 : SGPR spill to VGPR lane
; CHECK-NEXT:    s_mov_b64 s[4:5], 0
; CHECK-NEXT:    s_load_dwordx4 s[28:31], s[4:5], 0x0
; CHECK-NEXT:    s_movk_i32 s4, 0x130
; CHECK-NEXT:    s_mov_b32 s5, s24
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_writelane_b32 v4, s36, 0
; CHECK-NEXT:    v_writelane_b32 v4, s37, 1
; CHECK-NEXT:    v_writelane_b32 v4, s38, 2
; CHECK-NEXT:    v_writelane_b32 v4, s39, 3
; CHECK-NEXT:    v_writelane_b32 v4, s40, 4
; CHECK-NEXT:    v_writelane_b32 v4, s41, 5
; CHECK-NEXT:    v_writelane_b32 v4, s42, 6
; CHECK-NEXT:    v_writelane_b32 v4, s43, 7
; CHECK-NEXT:    v_writelane_b32 v4, s44, 8
; CHECK-NEXT:    v_writelane_b32 v4, s45, 9
; CHECK-NEXT:    v_writelane_b32 v4, s46, 10
; CHECK-NEXT:    s_load_dwordx16 s[4:19], s[4:5], 0x0
; CHECK-NEXT:    v_writelane_b32 v4, s47, 11
; CHECK-NEXT:    v_writelane_b32 v4, s48, 12
; CHECK-NEXT:    v_writelane_b32 v4, s49, 13
; CHECK-NEXT:    s_mov_b32 s20, 0
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_writelane_b32 v4, s50, 14
; CHECK-NEXT:    v_mov_b32_e32 v5, s28
; CHECK-NEXT:    v_mov_b32_e32 v6, v1
; CHECK-NEXT:    s_mov_b32 s21, s20
; CHECK-NEXT:    s_mov_b32 s22, s20
; CHECK-NEXT:    s_mov_b32 s23, s20
; CHECK-NEXT:    v_writelane_b32 v4, s51, 15
; CHECK-NEXT:    v_mov_b32_e32 v2, v1
; CHECK-NEXT:    image_sample_lz v5, v[5:6], s[44:51], s[20:23] dmask:0x1
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_writelane_b32 v4, s4, 16
; CHECK-NEXT:    v_writelane_b32 v4, s5, 17
; CHECK-NEXT:    v_writelane_b32 v4, s6, 18
; CHECK-NEXT:    v_writelane_b32 v4, s7, 19
; CHECK-NEXT:    v_writelane_b32 v4, s8, 20
; CHECK-NEXT:    v_writelane_b32 v4, s9, 21
; CHECK-NEXT:    image_sample_lz v6, v[1:2], s[4:11], s[20:23] dmask:0x1
; CHECK-NEXT:    v_writelane_b32 v4, s10, 22
; CHECK-NEXT:    v_writelane_b32 v4, s11, 23
; CHECK-NEXT:    v_writelane_b32 v4, s12, 24
; CHECK-NEXT:    v_writelane_b32 v4, s13, 25
; CHECK-NEXT:    v_writelane_b32 v4, s14, 26
; CHECK-NEXT:    v_writelane_b32 v4, s15, 27
; CHECK-NEXT:    v_writelane_b32 v4, s16, 28
; CHECK-NEXT:    v_writelane_b32 v8, s52, 18
; CHECK-NEXT:    v_writelane_b32 v4, s17, 29
; CHECK-NEXT:    v_writelane_b32 v8, s53, 19
; CHECK-NEXT:    v_writelane_b32 v4, s18, 30
; CHECK-NEXT:    v_writelane_b32 v8, s54, 20
; CHECK-NEXT:    v_writelane_b32 v4, s19, 31
; CHECK-NEXT:    s_mov_b32 s4, 48
; CHECK-NEXT:    s_mov_b32 s5, s24
; CHECK-NEXT:    v_writelane_b32 v8, s55, 21
; CHECK-NEXT:    s_load_dwordx8 s[4:11], s[4:5], 0x0
; CHECK-NEXT:    v_writelane_b32 v8, s56, 22
; CHECK-NEXT:    v_writelane_b32 v8, s57, 23
; CHECK-NEXT:    v_writelane_b32 v8, s58, 24
; CHECK-NEXT:    v_writelane_b32 v8, s59, 25
; CHECK-NEXT:    v_writelane_b32 v8, s60, 26
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_writelane_b32 v4, s4, 32
; CHECK-NEXT:    v_writelane_b32 v8, s61, 27
; CHECK-NEXT:    v_writelane_b32 v4, s5, 33
; CHECK-NEXT:    v_writelane_b32 v8, s62, 28
; CHECK-NEXT:    v_writelane_b32 v4, s6, 34
; CHECK-NEXT:    v_writelane_b32 v8, s63, 29
; CHECK-NEXT:    v_writelane_b32 v4, s7, 35
; CHECK-NEXT:    v_writelane_b32 v8, s64, 30
; CHECK-NEXT:    v_writelane_b32 v4, s8, 36
; CHECK-NEXT:    v_writelane_b32 v8, s65, 31
; CHECK-NEXT:    v_writelane_b32 v4, s9, 37
; CHECK-NEXT:    v_writelane_b32 v8, s66, 32
; CHECK-NEXT:    s_movk_i32 s26, 0x1f0
; CHECK-NEXT:    s_movk_i32 s28, 0x2f0
; CHECK-NEXT:    s_mov_b32 s27, s24
; CHECK-NEXT:    s_mov_b32 s29, s24
; CHECK-NEXT:    v_writelane_b32 v4, s10, 38
; CHECK-NEXT:    v_writelane_b32 v8, s67, 33
; CHECK-NEXT:    v_writelane_b32 v4, s11, 39
; CHECK-NEXT:    s_load_dwordx16 s[52:67], s[26:27], 0x0
; CHECK-NEXT:    s_load_dwordx16 s[4:19], s[28:29], 0x0
; CHECK-NEXT:    v_and_b32_e32 v0, 1, v0
; CHECK-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; CHECK-NEXT:    s_xor_b64 s[24:25], vcc, -1
; CHECK-NEXT:    ; implicit-def: $vgpr3 : SGPR spill to VGPR lane
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    v_mul_f32_e32 v0, v6, v5
; CHECK-NEXT:    s_and_saveexec_b64 s[26:27], s[24:25]
; CHECK-NEXT:    s_xor_b64 s[26:27], exec, s[26:27]
; CHECK-NEXT:    s_cbranch_execz .LBB0_3
; CHECK-NEXT:  ; %bb.1: ; %bb48
; CHECK-NEXT:    v_readlane_b32 s36, v4, 0
; CHECK-NEXT:    v_readlane_b32 s44, v4, 8
; CHECK-NEXT:    v_readlane_b32 s45, v4, 9
; CHECK-NEXT:    v_readlane_b32 s46, v4, 10
; CHECK-NEXT:    v_readlane_b32 s47, v4, 11
; CHECK-NEXT:    v_readlane_b32 s48, v4, 12
; CHECK-NEXT:    v_readlane_b32 s49, v4, 13
; CHECK-NEXT:    v_readlane_b32 s50, v4, 14
; CHECK-NEXT:    v_readlane_b32 s51, v4, 15
; CHECK-NEXT:    s_and_b64 vcc, exec, -1
; CHECK-NEXT:    v_readlane_b32 s37, v4, 1
; CHECK-NEXT:    v_readlane_b32 s38, v4, 2
; CHECK-NEXT:    v_readlane_b32 s39, v4, 3
; CHECK-NEXT:    v_readlane_b32 s40, v4, 4
; CHECK-NEXT:    image_sample_lz v5, v[1:2], s[44:51], s[20:23] dmask:0x1
; CHECK-NEXT:    v_mov_b32_e32 v2, 0
; CHECK-NEXT:    s_mov_b32 s21, s20
; CHECK-NEXT:    s_mov_b32 s22, s20
; CHECK-NEXT:    s_mov_b32 s23, s20
; CHECK-NEXT:    v_readlane_b32 s41, v4, 5
; CHECK-NEXT:    v_readlane_b32 s42, v4, 6
; CHECK-NEXT:    v_readlane_b32 s43, v4, 7
; CHECK-NEXT:  .LBB0_2: ; %bb50
; CHECK-NEXT:    ; =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    v_readlane_b32 s36, v4, 32
; CHECK-NEXT:    v_readlane_b32 s40, v4, 36
; CHECK-NEXT:    v_readlane_b32 s41, v4, 37
; CHECK-NEXT:    v_readlane_b32 s42, v4, 38
; CHECK-NEXT:    v_readlane_b32 s43, v4, 39
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    s_nop 3
; CHECK-NEXT:    image_sample_lz v6, v[1:2], s[60:67], s[40:43] dmask:0x1
; CHECK-NEXT:    s_nop 0
; CHECK-NEXT:    image_sample_lz v1, v[1:2], s[12:19], s[20:23] dmask:0x1
; CHECK-NEXT:    v_readlane_b32 s37, v4, 33
; CHECK-NEXT:    v_readlane_b32 s38, v4, 34
; CHECK-NEXT:    v_readlane_b32 s39, v4, 35
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    v_sub_f32_e32 v1, v1, v6
; CHECK-NEXT:    v_mul_f32_e32 v1, v1, v0
; CHECK-NEXT:    v_mul_f32_e32 v1, v1, v5
; CHECK-NEXT:    s_mov_b64 vcc, vcc
; CHECK-NEXT:    s_cbranch_vccnz .LBB0_2
; CHECK-NEXT:  .LBB0_3: ; %Flow14
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_readlane_b32 s12, v4, 32
; CHECK-NEXT:    v_readlane_b32 s13, v4, 33
; CHECK-NEXT:    v_readlane_b32 s14, v4, 34
; CHECK-NEXT:    v_readlane_b32 s15, v4, 35
; CHECK-NEXT:    v_readlane_b32 s16, v4, 36
; CHECK-NEXT:    v_readlane_b32 s17, v4, 37
; CHECK-NEXT:    v_readlane_b32 s18, v4, 38
; CHECK-NEXT:    v_readlane_b32 s19, v4, 39
; CHECK-NEXT:    v_writelane_b32 v4, s4, 40
; CHECK-NEXT:    v_writelane_b32 v4, s5, 41
; CHECK-NEXT:    v_writelane_b32 v4, s6, 42
; CHECK-NEXT:    v_writelane_b32 v4, s7, 43
; CHECK-NEXT:    v_writelane_b32 v4, s8, 44
; CHECK-NEXT:    v_writelane_b32 v4, s9, 45
; CHECK-NEXT:    v_writelane_b32 v4, s10, 46
; CHECK-NEXT:    v_writelane_b32 v4, s11, 47
; CHECK-NEXT:    v_writelane_b32 v4, s12, 48
; CHECK-NEXT:    v_writelane_b32 v4, s13, 49
; CHECK-NEXT:    v_writelane_b32 v4, s14, 50
; CHECK-NEXT:    v_writelane_b32 v4, s15, 51
; CHECK-NEXT:    v_writelane_b32 v4, s16, 52
; CHECK-NEXT:    v_writelane_b32 v4, s17, 53
; CHECK-NEXT:    v_writelane_b32 v4, s18, 54
; CHECK-NEXT:    v_writelane_b32 v4, s19, 55
; CHECK-NEXT:    v_writelane_b32 v4, s52, 56
; CHECK-NEXT:    v_writelane_b32 v3, s60, 0
; CHECK-NEXT:    v_writelane_b32 v4, s53, 57
; CHECK-NEXT:    v_writelane_b32 v3, s61, 1
; CHECK-NEXT:    v_writelane_b32 v4, s54, 58
; CHECK-NEXT:    v_writelane_b32 v3, s62, 2
; CHECK-NEXT:    v_writelane_b32 v4, s55, 59
; CHECK-NEXT:    v_writelane_b32 v3, s63, 3
; CHECK-NEXT:    v_writelane_b32 v4, s56, 60
; CHECK-NEXT:    v_writelane_b32 v3, s64, 4
; CHECK-NEXT:    v_writelane_b32 v4, s57, 61
; CHECK-NEXT:    v_writelane_b32 v3, s65, 5
; CHECK-NEXT:    v_writelane_b32 v4, s58, 62
; CHECK-NEXT:    v_writelane_b32 v3, s66, 6
; CHECK-NEXT:    v_writelane_b32 v4, s59, 63
; CHECK-NEXT:    v_writelane_b32 v3, s67, 7
; CHECK-NEXT:    s_andn2_saveexec_b64 s[20:21], s[26:27]
; CHECK-NEXT:    s_cbranch_execz .LBB0_10
; CHECK-NEXT:  ; %bb.4: ; %bb32
; CHECK-NEXT:    s_and_saveexec_b64 s[8:9], s[24:25]
; CHECK-NEXT:    s_xor_b64 s[22:23], exec, s[8:9]
; CHECK-NEXT:    s_cbranch_execz .LBB0_6
; CHECK-NEXT:  ; %bb.5: ; %bb43
; CHECK-NEXT:    s_mov_b32 s8, 0
; CHECK-NEXT:    s_mov_b32 s9, s8
; CHECK-NEXT:    v_mov_b32_e32 v0, s8
; CHECK-NEXT:    v_readlane_b32 s36, v4, 0
; CHECK-NEXT:    v_mov_b32_e32 v1, s9
; CHECK-NEXT:    s_mov_b32 s10, s8
; CHECK-NEXT:    s_mov_b32 s11, s8
; CHECK-NEXT:    v_readlane_b32 s37, v4, 1
; CHECK-NEXT:    v_readlane_b32 s38, v4, 2
; CHECK-NEXT:    v_readlane_b32 s39, v4, 3
; CHECK-NEXT:    v_readlane_b32 s40, v4, 4
; CHECK-NEXT:    v_readlane_b32 s41, v4, 5
; CHECK-NEXT:    v_readlane_b32 s42, v4, 6
; CHECK-NEXT:    v_readlane_b32 s43, v4, 7
; CHECK-NEXT:    v_readlane_b32 s44, v4, 8
; CHECK-NEXT:    v_readlane_b32 s45, v4, 9
; CHECK-NEXT:    v_readlane_b32 s46, v4, 10
; CHECK-NEXT:    v_readlane_b32 s47, v4, 11
; CHECK-NEXT:    v_readlane_b32 s48, v4, 12
; CHECK-NEXT:    v_readlane_b32 s49, v4, 13
; CHECK-NEXT:    v_readlane_b32 s50, v4, 14
; CHECK-NEXT:    v_readlane_b32 s51, v4, 15
; CHECK-NEXT:    image_sample_lz v5, v[0:1], s[36:43], s[8:11] dmask:0x1
; CHECK-NEXT:    v_readlane_b32 s36, v4, 16
; CHECK-NEXT:    v_readlane_b32 s44, v4, 24
; CHECK-NEXT:    v_readlane_b32 s45, v4, 25
; CHECK-NEXT:    v_readlane_b32 s46, v4, 26
; CHECK-NEXT:    v_readlane_b32 s47, v4, 27
; CHECK-NEXT:    v_readlane_b32 s48, v4, 28
; CHECK-NEXT:    v_readlane_b32 s49, v4, 29
; CHECK-NEXT:    v_readlane_b32 s50, v4, 30
; CHECK-NEXT:    v_readlane_b32 s51, v4, 31
; CHECK-NEXT:    v_mov_b32_e32 v6, 0
; CHECK-NEXT:    v_mov_b32_e32 v7, v6
; CHECK-NEXT:    v_readlane_b32 s37, v4, 17
; CHECK-NEXT:    v_readlane_b32 s38, v4, 18
; CHECK-NEXT:    v_readlane_b32 s39, v4, 19
; CHECK-NEXT:    image_sample_lz v0, v[0:1], s[44:51], s[12:15] dmask:0x1
; CHECK-NEXT:    v_readlane_b32 s40, v4, 20
; CHECK-NEXT:    v_readlane_b32 s41, v4, 21
; CHECK-NEXT:    v_readlane_b32 s42, v4, 22
; CHECK-NEXT:    v_readlane_b32 s43, v4, 23
; CHECK-NEXT:    s_waitcnt vmcnt(1)
; CHECK-NEXT:    buffer_store_dwordx3 v[5:7], off, s[8:11], 0
; CHECK-NEXT:    s_waitcnt vmcnt(1)
; CHECK-NEXT:    buffer_store_dwordx4 v[0:3], off, s[8:11], 0
; CHECK-NEXT:    ; implicit-def: $vgpr0
; CHECK-NEXT:  .LBB0_6: ; %Flow12
; CHECK-NEXT:    s_or_saveexec_b64 s[4:5], s[22:23]
; CHECK-NEXT:    v_readlane_b32 s52, v4, 40
; CHECK-NEXT:    v_readlane_b32 s53, v4, 41
; CHECK-NEXT:    v_readlane_b32 s54, v4, 42
; CHECK-NEXT:    v_readlane_b32 s55, v4, 43
; CHECK-NEXT:    v_readlane_b32 s56, v4, 44
; CHECK-NEXT:    v_readlane_b32 s57, v4, 45
; CHECK-NEXT:    v_readlane_b32 s58, v4, 46
; CHECK-NEXT:    v_readlane_b32 s59, v4, 47
; CHECK-NEXT:    v_readlane_b32 s60, v4, 48
; CHECK-NEXT:    v_readlane_b32 s61, v4, 49
; CHECK-NEXT:    v_readlane_b32 s62, v4, 50
; CHECK-NEXT:    v_readlane_b32 s63, v4, 51
; CHECK-NEXT:    v_readlane_b32 s64, v4, 52
; CHECK-NEXT:    v_readlane_b32 s65, v4, 53
; CHECK-NEXT:    v_readlane_b32 s66, v4, 54
; CHECK-NEXT:    v_readlane_b32 s67, v4, 55
; CHECK-NEXT:    s_xor_b64 exec, exec, s[4:5]
; CHECK-NEXT:    s_cbranch_execz .LBB0_9
; CHECK-NEXT:  ; %bb.7: ; %bb33.preheader
; CHECK-NEXT:    s_mov_b32 s8, 0
; CHECK-NEXT:    s_mov_b32 s6, s8
; CHECK-NEXT:    s_mov_b32 s7, s8
; CHECK-NEXT:    v_mov_b32_e32 v1, s6
; CHECK-NEXT:    v_readlane_b32 s36, v4, 56
; CHECK-NEXT:    s_mov_b32 s9, s8
; CHECK-NEXT:    s_mov_b32 s10, s8
; CHECK-NEXT:    s_mov_b32 s11, s8
; CHECK-NEXT:    v_mov_b32_e32 v2, s7
; CHECK-NEXT:    v_readlane_b32 s37, v4, 57
; CHECK-NEXT:    v_readlane_b32 s38, v4, 58
; CHECK-NEXT:    v_readlane_b32 s39, v4, 59
; CHECK-NEXT:    v_readlane_b32 s40, v4, 60
; CHECK-NEXT:    v_readlane_b32 s41, v4, 61
; CHECK-NEXT:    v_readlane_b32 s42, v4, 62
; CHECK-NEXT:    v_readlane_b32 s43, v4, 63
; CHECK-NEXT:    s_nop 4
; CHECK-NEXT:    image_sample_lz v5, v[1:2], s[36:43], s[8:11] dmask:0x1
; CHECK-NEXT:    image_sample_lz v6, v[1:2], s[52:59], s[8:11] dmask:0x1
; CHECK-NEXT:    ; kill: killed $vgpr1_vgpr2
; CHECK-NEXT:    s_mov_b64 s[12:13], s[36:37]
; CHECK-NEXT:    s_and_b64 vcc, exec, 0
; CHECK-NEXT:    v_readlane_b32 s44, v3, 0
; CHECK-NEXT:    v_readlane_b32 s45, v3, 1
; CHECK-NEXT:    v_readlane_b32 s46, v3, 2
; CHECK-NEXT:    v_readlane_b32 s47, v3, 3
; CHECK-NEXT:    v_readlane_b32 s48, v3, 4
; CHECK-NEXT:    v_readlane_b32 s49, v3, 5
; CHECK-NEXT:    v_readlane_b32 s50, v3, 6
; CHECK-NEXT:    v_readlane_b32 s51, v3, 7
; CHECK-NEXT:    s_mov_b64 s[14:15], s[38:39]
; CHECK-NEXT:    s_mov_b64 s[16:17], s[40:41]
; CHECK-NEXT:    s_mov_b64 s[18:19], s[42:43]
; CHECK-NEXT:    ; kill: killed $sgpr12_sgpr13_sgpr14_sgpr15_sgpr16_sgpr17_sgpr18_sgpr19
; CHECK-NEXT:    ; kill: killed $sgpr52_sgpr53_sgpr54_sgpr55_sgpr56_sgpr57_sgpr58_sgpr59
; CHECK-NEXT:    ; kill: killed $sgpr8_sgpr9_sgpr10 killed $sgpr11
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    v_sub_f32_e32 v1, v6, v5
; CHECK-NEXT:    v_mul_f32_e32 v0, v1, v0
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:  .LBB0_8: ; %bb33
; CHECK-NEXT:    ; =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    v_add_f32_e32 v2, v1, v0
; CHECK-NEXT:    v_sub_f32_e32 v1, v1, v2
; CHECK-NEXT:    s_mov_b64 vcc, vcc
; CHECK-NEXT:    s_cbranch_vccz .LBB0_8
; CHECK-NEXT:  .LBB0_9: ; %Flow13
; CHECK-NEXT:    s_or_b64 exec, exec, s[4:5]
; CHECK-NEXT:  .LBB0_10: ; %UnifiedReturnBlock
; CHECK-NEXT:    s_or_b64 exec, exec, s[20:21]
; CHECK-NEXT:    v_readlane_b32 s67, v8, 33
; CHECK-NEXT:    v_readlane_b32 s66, v8, 32
; CHECK-NEXT:    v_readlane_b32 s65, v8, 31
; CHECK-NEXT:    v_readlane_b32 s64, v8, 30
; CHECK-NEXT:    v_readlane_b32 s63, v8, 29
; CHECK-NEXT:    v_readlane_b32 s62, v8, 28
; CHECK-NEXT:    v_readlane_b32 s61, v8, 27
; CHECK-NEXT:    v_readlane_b32 s60, v8, 26
; CHECK-NEXT:    v_readlane_b32 s59, v8, 25
; CHECK-NEXT:    v_readlane_b32 s58, v8, 24
; CHECK-NEXT:    v_readlane_b32 s57, v8, 23
; CHECK-NEXT:    v_readlane_b32 s56, v8, 22
; CHECK-NEXT:    v_readlane_b32 s55, v8, 21
; CHECK-NEXT:    v_readlane_b32 s54, v8, 20
; CHECK-NEXT:    v_readlane_b32 s53, v8, 19
; CHECK-NEXT:    v_readlane_b32 s52, v8, 18
; CHECK-NEXT:    v_readlane_b32 s51, v8, 17
; CHECK-NEXT:    v_readlane_b32 s50, v8, 16
; CHECK-NEXT:    v_readlane_b32 s49, v8, 15
; CHECK-NEXT:    v_readlane_b32 s48, v8, 14
; CHECK-NEXT:    v_readlane_b32 s47, v8, 13
; CHECK-NEXT:    v_readlane_b32 s46, v8, 12
; CHECK-NEXT:    v_readlane_b32 s45, v8, 11
; CHECK-NEXT:    v_readlane_b32 s44, v8, 10
; CHECK-NEXT:    v_readlane_b32 s43, v8, 9
; CHECK-NEXT:    v_readlane_b32 s42, v8, 8
; CHECK-NEXT:    v_readlane_b32 s41, v8, 7
; CHECK-NEXT:    v_readlane_b32 s40, v8, 6
; CHECK-NEXT:    v_readlane_b32 s39, v8, 5
; CHECK-NEXT:    v_readlane_b32 s38, v8, 4
; CHECK-NEXT:    v_readlane_b32 s37, v8, 3
; CHECK-NEXT:    v_readlane_b32 s36, v8, 2
; CHECK-NEXT:    v_readlane_b32 s31, v8, 1
; CHECK-NEXT:    v_readlane_b32 s30, v8, 0
; CHECK-NEXT:    ; kill: killed $vgpr4
; CHECK-NEXT:    ; kill: killed $vgpr3
; CHECK-NEXT:    s_xor_saveexec_b64 s[4:5], -1
; CHECK-NEXT:    buffer_load_dword v8, off, s[0:3], s32 ; 4-byte Folded Reload
; CHECK-NEXT:    buffer_load_dword v4, off, s[0:3], s32 offset:4 ; 4-byte Folded Reload
; CHECK-NEXT:    buffer_load_dword v3, off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; CHECK-NEXT:    s_mov_b64 exec, s[4:5]
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
bb:
  %i = call i64 @llvm.amdgcn.s.getpc()
  %i1 = trunc i64 %i to i32
  %i2 = insertelement <2 x i32> zeroinitializer, i32 %i1, i64 1
  %i3 = bitcast <2 x i32> %i2 to i64
  %i4 = inttoptr i64 %i3 to ptr addrspace(4)
  %i5 = getelementptr i8, ptr addrspace(4) %i4, i64 48
  %i6 = load <4 x i32>, ptr addrspace(4) %i5, align 16
  %i7 = getelementptr i8, ptr addrspace(4) %i4, i64 64
  %i8 = load <4 x i32>, ptr addrspace(4) %i7, align 16
  %i9 = getelementptr i8, ptr addrspace(4) %i4, i64 240
  %i10 = load <8 x i32>, ptr addrspace(4) %i9, align 32
  %i11 = getelementptr i8, ptr addrspace(4) %i4, i64 272
  %i12 = load <8 x i32>, ptr addrspace(4) %i11, align 32
  %i13 = getelementptr i8, ptr addrspace(4) %i4, i64 304
  %i14 = load <8 x i32>, ptr addrspace(4) %i13, align 32
  %i15 = getelementptr i8, ptr addrspace(4) %i4, i64 336
  %i16 = load <8 x i32>, ptr addrspace(4) %i15, align 32
  %i17 = getelementptr i8, ptr addrspace(4) %i4, i64 496
  %i18 = load <8 x i32>, ptr addrspace(4) %i17, align 32
  %i19 = getelementptr i8, ptr addrspace(4) %i4, i64 528
  %i20 = load <8 x i32>, ptr addrspace(4) %i19, align 32
  %i21 = getelementptr i8, ptr addrspace(4) %i4, i64 752
  %i22 = load <8 x i32>, ptr addrspace(4) %i21, align 32
  %i23 = getelementptr i8, ptr addrspace(4) %i4, i64 784
  %i24 = load <8 x i32>, ptr addrspace(4) %i23, align 32
  %i25 = load <4 x float>, ptr addrspace(4) null, align 16
  %i26 = extractelement <4 x float> %i25, i64 0
  %i27 = call float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 1, float %i26, float 0.000000e+00, <8 x i32> %i12, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  %i28 = call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 1, float 0.000000e+00, float 0.000000e+00, <8 x i32> %i14, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  %i29 = extractelement <4 x float> %i28, i64 0
  %i30 = fmul float %i29, %i27
  %i31 = call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 1, float 0.000000e+00, float 0.000000e+00, <8 x i32> %i16, <4 x i32> %i6, i1 false, i32 0, i32 0)
  br i1 %arg, label %bb32, label %bb48

bb32:                                             ; preds = %bb
  br i1 %arg, label %bb33, label %bb43

bb33:                                             ; preds = %bb33, %bb32
  %i34 = phi float [ %i42, %bb33 ], [ 0.000000e+00, %bb32 ]
  %i35 = call <2 x float> @llvm.amdgcn.image.sample.lz.2d.v2f32.f32(i32 1, float 0.000000e+00, float 0.000000e+00, <8 x i32> %i18, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  %i36 = extractelement <2 x float> %i35, i64 0
  %i37 = call <2 x float> @llvm.amdgcn.image.sample.lz.2d.v2f32.f32(i32 1, float 0.000000e+00, float 0.000000e+00, <8 x i32> %i22, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  %i38 = extractelement <2 x float> %i37, i64 0
  %i39 = fsub float %i38, %i36
  %i40 = fmul float %i39, %i30
  %i41 = fadd float %i34, %i40
  %i42 = fsub float %i34, %i41
  br label %bb33

bb43:                                             ; preds = %bb32
  %i44 = call float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 1, float 0.000000e+00, float 0.000000e+00, <8 x i32> %i10, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  %i45 = bitcast float %i44 to i32
  %i46 = insertelement <3 x i32> zeroinitializer, i32 %i45, i64 0
  call void @llvm.amdgcn.raw.buffer.store.v3i32(<3 x i32> %i46, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  %i47 = bitcast <4 x float> %i31 to <4 x i32>
  call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %i47, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  ret void

bb48:                                             ; preds = %bb
  %i49 = call float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 1, float 0.000000e+00, float 0.000000e+00, <8 x i32> %i12, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  br label %bb50

bb50:                                             ; preds = %bb50, %bb48
  %i51 = phi float [ 0.000000e+00, %bb48 ], [ %i58, %bb50 ]
  %i52 = call <2 x float> @llvm.amdgcn.image.sample.lz.2d.v2f32.f32(i32 1, float %i51, float 0.000000e+00, <8 x i32> %i20, <4 x i32> %i8, i1 false, i32 0, i32 0)
  %i53 = extractelement <2 x float> %i52, i64 0
  %i54 = call <2 x float> @llvm.amdgcn.image.sample.lz.2d.v2f32.f32(i32 1, float %i51, float 0.000000e+00, <8 x i32> %i24, <4 x i32> zeroinitializer, i1 false, i32 0, i32 0)
  %i55 = extractelement <2 x float> %i54, i64 0
  %i56 = fsub float %i55, %i53
  %i57 = fmul float %i56, %i30
  %i58 = fmul float %i57, %i49
  br label %bb50
}

declare i64 @llvm.amdgcn.s.getpc() #1
declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #2
declare float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #2
declare <2 x float> @llvm.amdgcn.image.sample.lz.2d.v2f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #2
declare void @llvm.amdgcn.raw.buffer.store.v3i32(<3 x i32>, <4 x i32>, i32, i32, i32 immarg) #3
declare void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32 immarg) #3

attributes #0 = { "amdgpu-waves-per-eu"="10,10" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(write) }
