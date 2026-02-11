# RLVR: Reinforcement Learning with Verifiable Rewards
# For ZK-ML compatible integer-only models

from .rewards import (
    VerifiableReward,
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

__all__ = [
    "VerifiableReward",
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
]
