//=== lib/CodeGen/GlobalISel/AIE2PreLegalizerCombiner.cpp --------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// before the legalizer.
//
//===----------------------------------------------------------------------===//

#include "AIE2TargetMachine.h"
#include "AIECombinerHelper.h"
#include "MCTargetDesc/AIE2MCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "aie2-prelegalizer-combiner"

#define GET_GICOMBINER_DEPS
#include "AIE2GenPreLegalizerGICombiner.inc"
#undef GET_GICOMBINER_DEPS

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "AIE2GenPreLegalizerGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class AIE2PreLegalizerCombinerImpl : public Combiner {
protected:
  // TODO: Make CombinerHelper methods const.
  mutable CombinerHelper Helper;
  const AIE2PreLegalizerCombinerImplRuleConfig &RuleConfig;
  const AIE2Subtarget &STI;

public:
  AIE2PreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelKnownBits &KB, GISelCSEInfo *CSEInfo,
      const AIE2PreLegalizerCombinerImplRuleConfig &RuleConfig,
      const AIE2Subtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "AIE2PreLegalizerCombiner"; }
  bool tryCombineAll(MachineInstr &I) const override;
  bool tryCombineAllImpl(MachineInstr &I) const;
  bool tryCombineShuffleVector(MachineInstr &MI) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "AIE2GenPreLegalizerGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "AIE2GenPreLegalizerGICombiner.inc"
#undef GET_GICOMBINER_IMPL

AIE2PreLegalizerCombinerImpl::AIE2PreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelKnownBits &KB, GISelCSEInfo *CSEInfo,
    const AIE2PreLegalizerCombinerImplRuleConfig &RuleConfig,
    const AIE2Subtarget &STI,
    MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &KB, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ false, &KB, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "AIE2GenPreLegalizerGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

std::function<std::optional<int32_t>()>
sectionGenerator(const int32_t From, const int32_t To,
                 const int32_t Partitions) {
  int32_t RoundSize = To / Partitions;
  int32_t Index = From;
  return [RoundSize, Index, Partitions, To]() mutable {
    int32_t Next = (Index / Partitions) + RoundSize * (Index % Partitions) +
                   (Index % Partitions);
    std::optional<int32_t> Return = std::optional<int32_t>(Next);
    if (Index++ == To)
      Return = {};
    return Return;
  };
}

bool AIE2PreLegalizerCombinerImpl::tryCombineShuffleVector(
    MachineInstr &MI) const {
  const Register DstReg = MI.getOperand(0).getReg();
  const LLT DstTy = MRI.getType(DstReg);
  const LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
  const unsigned DstNumElts = DstTy.isVector() ? DstTy.getNumElements() : 1;
  const unsigned SrcNumElts = SrcTy.isVector() ? SrcTy.getNumElements() : 1;
  MachineIRBuilder MIB(MI);
  MachineRegisterInfo &MRI = *MIB.getMRI();

  if (Helper.tryCombineShuffleVector(MI))
    return true;

  std::function<std::optional<int32_t>()> FourPartitions =
      sectionGenerator(0, DstNumElts - 1, 4);
  if (Helper.matchCombineShuffleVectorSimple(MI, FourPartitions, DstNumElts)) {
    const Register Src1 = MI.getOperand(1).getReg();
    const Register Src2 = MI.getOperand(2).getReg();
    const Register ShuffleModeReg =
        MRI.createGenericVirtualRegister(LLT::scalar(32));

    MIB.buildConstant(ShuffleModeReg, 29);
    MIB.buildInstr(AIE2::G_AIE_VSHUFFLE, {DstReg},
                   {Src1, Src2, ShuffleModeReg});
    MI.eraseFromParent();
    return true;
  }
  return false;
}
bool AIE2PreLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  if (tryCombineAllImpl(MI))
    return true;

  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return tryCombineShuffleVector(MI);
  }
  return false;
}

class AIE2PreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AIE2PreLegalizerCombiner();

  StringRef getPassName() const override { return "AIE2PreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
    getSelectionDAGFallbackAnalysisUsage(AU);
    AU.addRequired<GISelKnownBitsAnalysis>();
    AU.addPreserved<GISelKnownBitsAnalysis>();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addPreserved<GISelCSEAnalysisWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  AIE2PreLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

AIE2PreLegalizerCombiner::AIE2PreLegalizerCombiner()
    : MachineFunctionPass(ID) {
  initializeAIE2PreLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AIE2PreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();

  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC->getCSEConfig());

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  const AIE2Subtarget &ST = MF.getSubtarget<AIE2Subtarget>();
  const auto *LI = ST.getLegalizerInfo();

  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();

  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());
  AIE2PreLegalizerCombinerImpl Impl(MF, CInfo, TPC, *KB, CSEInfo,
                                        RuleConfig, ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char AIE2PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AIE2PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine AIE2 machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(AIE2PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine AIE2 machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createAIE2PreLegalizerCombiner() {
  return new AIE2PreLegalizerCombiner();
}
} // end namespace llvm
