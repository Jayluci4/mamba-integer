# RLVR: Reinforcement Learning with Verifiable Rewards
# For ZK-ML compatible integer-only models

from .rewards import (
    VerifiableReward,
    RewardResult,
    ArithmeticReward,
    CodeExecutionReward,
    LogicReward,
    FormatReward,
    CompositeReward,
)
from .grpo import GRPOTrainer, GRPOConfig, MathProblemDataset
from .zk_rewards import (
    ZKVerifiableReward,
    IntegerArithmeticProof,
    ZKIntegerArithmeticReward,
    ZKDyadicRationalReward,
    ZKCompositeReward,
)
from .domain_verifiers import (
    SolidityVerifier,
    SQLVerifier,
    MathVerifier,
    SECFinanceVerifier,
    EnglishVerifier,
    IVGRPOReward,
    create_verifier,
    create_iv_grpo_verifier,
)
from .iv_grpo import (
    IVGRPOConfig,
    IVGRPOTrainer,
    DomainDataset,
    MultiDomainIVGRPOTrainer,
)
from .sft_generator import (
    SFTGenerationConfig,
    SFTGenerator,
    SFTGenerationResult,
)

__all__ = [
    "VerifiableReward",
    "RewardResult",
    "ArithmeticReward",
    "CodeExecutionReward",
    "LogicReward",
    "FormatReward",
    "CompositeReward",
    "GRPOTrainer",
    "GRPOConfig",
    "MathProblemDataset",
    "ZKVerifiableReward",
    "IntegerArithmeticProof",
    "ZKIntegerArithmeticReward",
    "ZKDyadicRationalReward",
    "ZKCompositeReward",
    "SolidityVerifier",
    "SQLVerifier",
    "MathVerifier",
    "SECFinanceVerifier",
    "EnglishVerifier",
    "IVGRPOReward",
    "create_verifier",
    "create_iv_grpo_verifier",
    "IVGRPOConfig",
    "IVGRPOTrainer",
    "DomainDataset",
    "MultiDomainIVGRPOTrainer",
    "SFTGenerationConfig",
    "SFTGenerator",
    "SFTGenerationResult",
]
