//===-- AIE2AsmPrinter.cpp - AIEngine V2 LLVM assembly writer ------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the AIEngine V2 assembly language.
//
//===----------------------------------------------------------------------===//

#include "AIE2AsmPrinter.h"
#include "AIE2TargetMachine.h"
#include "InstPrinter/AIE2InstPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "AIE2GenMCPseudoLowering.inc"

bool AIE2AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                     const char *ExtraCode, raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << AIE2InstPrinter::getRegisterName(MO.getReg());
    return false;
  default:
    break;
  }

  return true;
}

bool AIE2AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                           unsigned OpNo, const char *ExtraCode,
                                           raw_ostream &OS) {
  if (!ExtraCode) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // For now, we only support register memory operands in registers and
    // assume there is no addend
    if (!MO.isReg())
      return true;

    OS << "0(" << AIE2InstPrinter::getRegisterName(MO.getReg()) << ")";
    return false;
  }

  return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);
}

bool AIE2AsmPrinter::lowerOperand(const MachineOperand &MO,
                                  MCOperand &MCOp) const {
  return LowerAIEMachineOperandToMCOperand(MO, MCOp, *this);
}

void AIE2AsmPrinter::emitInstruction(const MachineInstr *MI) {
  // PseudoLoopEnd is just here to carry some extra information. No need to
  // emit anything
  if (MI->getOpcode() == AIE2::PseudoLoopEnd)
    return;

  // Check whether there's a PseudoLoopEnd following. If so, emit the
  // last-bundle-label that it carries to designate this MI as the last bundle.
  const MachineInstr *LastBundle = nullptr;
  const MachineInstr *LoopEnd = nullptr;
  const MachineInstr *Prev = nullptr;
  for (auto &MII : *MI->getParent()) {
    if (MII.getOpcode() == AIE2::PseudoLoopEnd) {
      LoopEnd = &MII;
      LastBundle = Prev;
      break;
    }
    Prev = &MII;
  }

  if (MI == LastBundle) {
    MCSymbol *S = LoopEnd->getOperand(1).getMCSymbol();
    OutStreamer->emitLabel(S);
    OutStreamer->getCommentOS() << "Loop end.\n";
  }
  AIEBaseAsmPrinter::emitInstruction(MI);
}

AsmPrinter *
llvm::createAIE2AsmPrinterPass(TargetMachine &TM,
                               std::unique_ptr<MCStreamer> &&Streamer) {
  return new AIE2AsmPrinter(TM, std::move(Streamer));
}
