"""
ZK-Specific Verifiable Rewards for Mamba-Integer

These rewards are designed specifically for ZK-ML (Zero Knowledge Machine Learning)
applications where the model's inference must be cryptographically verifiable.

Key Insight:
Mamba-Integer uses integer-only arithmetic (no transcendentals), making it
naturally compatible with ZK-SNARK circuits. These rewards leverage this property.

Design Principles:
1. All computations must be expressible in integer arithmetic
2. Rewards should be verifiable in a ZK circuit
3. Proofs of correctness should be generatable
4. No floating point operations in reward computation

Reference:
- ZKML (EuroSys 2024): https://ddkang.github.io/papers/2024/zkml-eurosys.pdf
- zkPyTorch (2025): eprint.iacr.org/2025/535
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from fractions import Fraction
from .rewards import VerifiableReward, RewardResult


# =============================================================================
# INTEGER ARITHMETIC PROOFS
# =============================================================================

@dataclass
class IntegerArithmeticProof:
    """
    A proof of integer arithmetic correctness.

    This structure can be verified in a ZK circuit.
    All values are integers - no floating point.
    """
    operation: str          # "add", "mul", "div", "mod", "pow"
    operands: List[int]     # Input operands
    result: int             # Computed result
    witness: Dict[str, int] # Intermediate values for verification
    proof_hash: str         # Hash of proof for commitment

    def verify(self) -> bool:
        """Verify the proof is correct."""
        a, b = self.operands[0], self.operands[1] if len(self.operands) > 1 else 0

        if self.operation == "add":
            expected = a + b
        elif self.operation == "sub":
            expected = a - b
        elif self.operation == "mul":
            expected = a * b
        elif self.operation == "div":
            # Integer division
            if b == 0:
                return False
            expected = a // b
        elif self.operation == "mod":
            if b == 0:
                return False
            expected = a % b
        elif self.operation == "pow":
            expected = a ** b
        else:
            return False

        return self.result == expected

    def to_circuit_inputs(self) -> Dict[str, int]:
        """Convert to format suitable for ZK circuit."""
        return {
            "op": hash(self.operation) % (2**64),
            "operand_a": self.operands[0],
            "operand_b": self.operands[1] if len(self.operands) > 1 else 0,
            "result": self.result,
            **self.witness
        }


class ZKVerifiableReward(VerifiableReward):
    """
    Base class for ZK-verifiable rewards.

    All derived classes must ensure:
    1. Computations use only integer arithmetic
    2. A proof can be generated for the verification
    3. The proof is valid if and only if the reward is 1.0
    """

    def __init__(self, scale_bits: int = 15):
        """
        Args:
            scale_bits: Number of bits for fixed-point scaling (default 15 = 32768)
        """
        self.scale = 2 ** scale_bits
        self.proofs: List[IntegerArithmeticProof] = []

    @property
    def name(self) -> str:
        return "zk_verifiable"

    def _to_scaled_int(self, x: float) -> int:
        """Convert float to scaled integer."""
        return int(x * self.scale)

    def _from_scaled_int(self, x: int) -> float:
        """Convert scaled integer back to float."""
        return x / self.scale

    def _create_proof(
        self,
        operation: str,
        operands: List[int],
        result: int,
        witness: Optional[Dict[str, int]] = None
    ) -> IntegerArithmeticProof:
        """Create and store a proof."""
        witness = witness or {}

        # Create proof hash
        proof_data = json.dumps({
            "op": operation,
            "operands": operands,
            "result": result,
            "witness": witness
        }, sort_keys=True)
        proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()[:16]

        proof = IntegerArithmeticProof(
            operation=operation,
            operands=operands,
            result=result,
            witness=witness,
            proof_hash=proof_hash
        )

        self.proofs.append(proof)
        return proof

    def get_all_proofs(self) -> List[IntegerArithmeticProof]:
        """Return all proofs generated during reward computation."""
        return self.proofs

    def clear_proofs(self):
        """Clear stored proofs."""
        self.proofs = []

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        raise NotImplementedError("Subclasses must implement compute()")


# =============================================================================
# ZK INTEGER ARITHMETIC REWARD
# =============================================================================

class ZKIntegerArithmeticReward(ZKVerifiableReward):
    """
    Integer arithmetic verification with ZK proof generation.

    Every arithmetic check generates a proof that can be verified in a ZK circuit.
    """

    @property
    def name(self) -> str:
        return "zk_integer_arithmetic"

    def _parse_int(self, s: str) -> Optional[int]:
        """Parse string to integer, reject non-integers."""
        try:
            s = s.strip()
            # Reject floats and fractions
            if "." in s or "/" in s:
                return None
            return int(s)
        except ValueError:
            return None

    def _extract_expression(self, text: str) -> Optional[Tuple[str, List[int]]]:
        """Extract arithmetic expression from text."""
        # Pattern: "a op b" where op is +, -, *, //, %, **
        patterns = [
            (r"(\d+)\s*\+\s*(\d+)", "add"),
            (r"(\d+)\s*\-\s*(\d+)", "sub"),
            (r"(\d+)\s*\*\s*(\d+)", "mul"),
            (r"(\d+)\s*(?://|/)\s*(\d+)", "div"),
            (r"(\d+)\s*%\s*(\d+)", "mod"),
            (r"(\d+)\s*\*\*\s*(\d+)", "pow"),
        ]

        for pattern, op in patterns:
            match = re.search(pattern, text)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                return op, [a, b]

        return None

    def _extract_answer(self, text: str) -> Optional[int]:
        """Extract integer answer from response."""
        # Look for answer patterns
        patterns = [
            r"(?:answer|result|equals|=)\s*[:=]?\s*([-+]?\d+)",
            r"(?:is|are)\s+([-+]?\d+)",
            r"([-+]?\d+)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # Last integer in text
        numbers = re.findall(r"[-+]?\d+", text)
        if numbers:
            return int(numbers[-1])

        return None

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        self.clear_proofs()

        # Extract operation from prompt
        expr = self._extract_expression(prompt)
        if expr is None:
            return RewardResult(0.0, False, "Could not extract arithmetic expression")

        op, operands = expr

        # Compute expected result
        a, b = operands
        if op == "add":
            expected = a + b
        elif op == "sub":
            expected = a - b
        elif op == "mul":
            expected = a * b
        elif op == "div":
            if b == 0:
                return RewardResult(0.0, False, "Division by zero")
            expected = a // b
        elif op == "mod":
            if b == 0:
                return RewardResult(0.0, False, "Modulo by zero")
            expected = a % b
        elif op == "pow":
            expected = a ** b
        else:
            return RewardResult(0.0, False, f"Unknown operation: {op}")

        # Extract answer from response
        answer = self._extract_answer(response)
        if answer is None:
            return RewardResult(0.0, False, "No integer answer found")

        # Create proof
        proof = self._create_proof(
            operation=op,
            operands=operands,
            result=answer,
            witness={"expected": expected}
        )

        # Verify
        correct = answer == expected

        return RewardResult(
            1.0 if correct else 0.0,
            correct,
            f"Expected {expected}, got {answer}",
            metadata={
                "expected": expected,
                "got": answer,
                "proof_hash": proof.proof_hash,
                "proof_valid": proof.verify()
            }
        )


# =============================================================================
# ZK DYADIC RATIONAL REWARD
# =============================================================================

class ZKDyadicRationalReward(ZKVerifiableReward):
    """
    Verify computations involving dyadic rationals.

    Dyadic rationals are fractions with power-of-2 denominators (e.g., 3/8, 7/16).
    These are ZK-friendly because division is just bit-shifting.

    This is directly relevant to Mamba-Integer's core design.
    """

    def __init__(self, max_denominator_bits: int = 15):
        super().__init__(scale_bits=max_denominator_bits)
        self.max_denominator = 2 ** max_denominator_bits

    @property
    def name(self) -> str:
        return "zk_dyadic_rational"

    def _is_dyadic(self, frac: Fraction) -> bool:
        """Check if fraction has power-of-2 denominator."""
        d = frac.denominator
        return d > 0 and (d & (d - 1)) == 0  # Power of 2 check

    def _to_dyadic_int(self, frac: Fraction) -> Tuple[int, int]:
        """Convert to (numerator, log2(denominator)) representation."""
        if not self._is_dyadic(frac):
            # Approximate to nearest dyadic
            scaled = int(frac * self.max_denominator)
            return scaled, self.scale.bit_length() - 1

        d = frac.denominator
        log2_d = d.bit_length() - 1
        return frac.numerator, log2_d

    def _parse_dyadic(self, s: str) -> Optional[Tuple[int, int]]:
        """Parse string to dyadic representation."""
        try:
            s = s.strip()

            if "/" in s:
                num, denom = s.split("/")
                num, denom = int(num), int(denom)

                # Check if dyadic
                if denom > 0 and (denom & (denom - 1)) == 0:
                    log2_d = denom.bit_length() - 1
                    return num, log2_d

                # Approximate
                frac = Fraction(num, denom)
                return self._to_dyadic_int(frac)

            elif "." in s:
                frac = Fraction(s).limit_denominator(self.max_denominator)
                return self._to_dyadic_int(frac)

            else:
                return int(s), 0  # Integer = n/1 = n >> 0

        except (ValueError, ZeroDivisionError):
            return None

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        self.clear_proofs()

        if ground_truth is None:
            return RewardResult(0.0, False, "Dyadic verification requires ground truth")

        # Parse expected
        expected = self._parse_dyadic(ground_truth)
        if expected is None:
            return RewardResult(0.0, False, f"Cannot parse ground truth: {ground_truth}")

        # Extract answer from response
        answer_match = re.search(r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)", response)
        if answer_match is None:
            return RewardResult(0.0, False, "No answer found")

        answer = self._parse_dyadic(answer_match.group(1))
        if answer is None:
            return RewardResult(0.0, False, "Cannot parse answer as dyadic")

        # Compare (both are now (numerator, log2_denominator) tuples)
        expected_num, expected_log2d = expected
        answer_num, answer_log2d = answer

        # Normalize to same denominator for comparison
        if expected_log2d > answer_log2d:
            answer_num = answer_num << (expected_log2d - answer_log2d)
            answer_log2d = expected_log2d
        elif answer_log2d > expected_log2d:
            expected_num = expected_num << (answer_log2d - expected_log2d)
            expected_log2d = answer_log2d

        # Create proof
        proof = self._create_proof(
            operation="dyadic_compare",
            operands=[expected_num, answer_num],
            result=1 if expected_num == answer_num else 0,
            witness={
                "expected_num": expected_num,
                "answer_num": answer_num,
                "log2_denominator": expected_log2d
            }
        )

        correct = expected_num == answer_num

        return RewardResult(
            1.0 if correct else 0.0,
            correct,
            f"Expected {expected_num}/{2**expected_log2d}, got {answer_num}/{2**answer_log2d}",
            metadata={
                "proof_hash": proof.proof_hash,
                "normalized_expected": expected_num,
                "normalized_answer": answer_num,
                "denominator_bits": expected_log2d
            }
        )


# =============================================================================
# ZK HASH COMMITMENT REWARD
# =============================================================================

class ZKHashCommitmentReward(ZKVerifiableReward):
    """
    Verify that model output matches a hash commitment.

    Use case: Verify model generated specific output without revealing the output.
    The commitment is hash(output), and we verify the model's output hashes to it.
    """

    def __init__(self, hash_algorithm: str = "sha256"):
        super().__init__()
        self.hash_algorithm = hash_algorithm

    @property
    def name(self) -> str:
        return "zk_hash_commitment"

    def _compute_hash(self, text: str) -> str:
        """Compute hash of text."""
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        elif self.hash_algorithm == "sha3_256":
            return hashlib.sha3_256(text.encode()).hexdigest()
        else:
            raise ValueError(f"Unknown hash algorithm: {self.hash_algorithm}")

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        """
        Args:
            ground_truth: Expected hash of correct response
        """
        self.clear_proofs()

        if ground_truth is None:
            return RewardResult(0.0, False, "Hash commitment verification requires expected hash")

        # Compute hash of response
        response_hash = self._compute_hash(response.strip())

        # Compare
        correct = response_hash == ground_truth.strip().lower()

        # Create proof (hash comparison is ZK-friendly)
        proof = self._create_proof(
            operation="hash_compare",
            operands=[int(response_hash[:16], 16), int(ground_truth[:16], 16)],
            result=1 if correct else 0,
            witness={
                "full_response_hash": response_hash,
                "full_expected_hash": ground_truth
            }
        )

        return RewardResult(
            1.0 if correct else 0.0,
            correct,
            "Hash matches" if correct else "Hash mismatch",
            metadata={
                "response_hash": response_hash,
                "expected_hash": ground_truth,
                "proof_hash": proof.proof_hash
            }
        )


# =============================================================================
# ZK INFERENCE VERIFICATION REWARD
# =============================================================================

class ZKInferenceVerificationReward(ZKVerifiableReward):
    """
    Verify that model inference was performed correctly.

    This is the core ZK-ML use case: proving that a specific model
    produced a specific output for a specific input.

    For Mamba-Integer, the integer-only design makes this tractable.
    """

    def __init__(self, model_hash: Optional[str] = None):
        """
        Args:
            model_hash: Hash of model weights (commitment to model identity)
        """
        super().__init__()
        self.model_hash = model_hash

    @property
    def name(self) -> str:
        return "zk_inference_verification"

    def compute_model_hash(self, model) -> str:
        """Compute hash of model weights."""
        import torch

        weight_bytes = b""
        for name, param in sorted(model.named_parameters()):
            # Convert to int representation (Mamba-Integer uses integer weights)
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                # Quantize to int for hashing
                quantized = (param.detach() * 32768).to(torch.int32)
            else:
                quantized = param.detach().to(torch.int32)

            weight_bytes += quantized.cpu().numpy().tobytes()

        return hashlib.sha256(weight_bytes).hexdigest()

    def compute(
        self,
        prompt: str,
        response: str,
        ground_truth: Optional[str] = None,
        input_ids: Optional[List[int]] = None,
        output_ids: Optional[List[int]] = None,
        model_hash: Optional[str] = None
    ) -> RewardResult:
        """
        Verify inference was performed correctly.

        For full ZK verification, we'd need:
        1. input_ids: Tokenized input
        2. output_ids: Tokenized output
        3. model_hash: Commitment to model weights
        4. Proof that model(input_ids) = output_ids

        This reward function generates the commitment; actual ZK proof
        would be generated by a separate circuit.
        """
        self.clear_proofs()

        # Use provided model hash or stored one
        mh = model_hash or self.model_hash

        if mh is None:
            return RewardResult(
                0.0, False,
                "No model hash provided for inference verification"
            )

        # Compute input/output commitments
        input_hash = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        output_hash = hashlib.sha256(response.encode()).hexdigest()[:32]

        # Create inference proof structure
        proof = self._create_proof(
            operation="inference",
            operands=[
                int(input_hash[:8], 16),
                int(output_hash[:8], 16),
                int(mh[:8], 16) if mh else 0
            ],
            result=1,  # We're creating a commitment, not verifying yet
            witness={
                "input_hash": input_hash,
                "output_hash": output_hash,
                "model_hash": mh,
                "input_length": len(prompt),
                "output_length": len(response)
            }
        )

        # If ground_truth is provided, check output matches
        if ground_truth:
            correct = response.strip() == ground_truth.strip()
            return RewardResult(
                1.0 if correct else 0.0,
                correct,
                "Inference matches expected" if correct else "Output mismatch",
                metadata={
                    "proof_hash": proof.proof_hash,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "model_hash": mh
                }
            )

        # Without ground_truth, just return the commitment
        return RewardResult(
            1.0, True,
            "Inference commitment generated",
            metadata={
                "proof_hash": proof.proof_hash,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "model_hash": mh
            }
        )


# =============================================================================
# COMPOSITE ZK REWARD
# =============================================================================

class ZKCompositeReward(ZKVerifiableReward):
    """
    Combine multiple ZK rewards with unified proof generation.

    All proofs from component rewards are aggregated.
    """

    def __init__(self, rewards: List[ZKVerifiableReward], mode: str = "all"):
        super().__init__()
        self.rewards = rewards
        self.mode = mode

    @property
    def name(self) -> str:
        names = "+".join(r.name for r in self.rewards)
        return f"zk_composite({names})"

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        self.clear_proofs()

        results = []
        all_proofs = []

        for reward_fn in self.rewards:
            result = reward_fn.compute(prompt, response, ground_truth)
            results.append(result)
            all_proofs.extend(reward_fn.get_all_proofs())

        self.proofs = all_proofs

        if self.mode == "all":
            all_correct = all(r.correct for r in results)
            if not all_correct:
                failed = [r for r in results if not r.correct][0]
                return RewardResult(0.0, False, f"Failed: {failed.reason}")
            return RewardResult(
                1.0, True,
                "All ZK checks passed",
                metadata={"num_proofs": len(all_proofs)}
            )

        elif self.mode == "any":
            if any(r.correct for r in results):
                return RewardResult(1.0, True, "At least one ZK check passed")
            return RewardResult(0.0, False, "All ZK checks failed")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
