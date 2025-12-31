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
from .grpo import GRPOTrainer
from .zk_rewards import ZKVerifiableReward, IntegerArithmeticProof

__all__ = [
    "VerifiableReward",
    "ArithmeticReward",
    "CodeExecutionReward",
    "LogicReward",
    "FormatReward",
    "CompositeReward",
    "GRPOTrainer",
    "ZKVerifiableReward",
    "IntegerArithmeticProof",
]
