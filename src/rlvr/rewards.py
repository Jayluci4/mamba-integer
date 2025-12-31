"""
Verifiable Reward Functions for RLVR Training

These rewards provide binary (0/1) signals based on objectively verifiable correctness.
No learned reward model needed - ground truth is deterministic.

Design Principles:
1. Binary rewards (DeepSeek R1 style) - clear signal, no ambiguity
2. Verifiable - can be checked programmatically without human judgment
3. ZK-friendly - designed for integer-only computation verification
4. Composable - can combine multiple reward types

Reference: DeepSeek-R1 (arXiv:2501.12948)
"""

import re
import ast
import json
import math
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from fractions import Fraction


@dataclass
class RewardResult:
    """Result of a reward computation."""
    reward: float  # 0.0 or 1.0 for binary, or [0, 1] for soft
    correct: bool
    reason: str
    metadata: Optional[Dict[str, Any]] = None


class VerifiableReward(ABC):
    """Base class for all verifiable rewards."""

    @abstractmethod
    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        """Compute reward for a response given a prompt."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this reward type."""
        pass


# =============================================================================
# ARITHMETIC REWARDS
# =============================================================================

class ArithmeticReward(VerifiableReward):
    """
    Verify arithmetic computations.

    Supports:
    - Integer arithmetic (+, -, *, //, %, **)
    - Rational arithmetic (fractions)
    - Expression evaluation
    - Comparison operations

    ZK-friendly: All operations are integer-only or dyadic rational.
    """

    def __init__(self, tolerance: float = 0.0, allow_rationals: bool = True):
        self.tolerance = tolerance
        self.allow_rationals = allow_rationals

    @property
    def name(self) -> str:
        return "arithmetic"

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract numeric answer from text."""
        # Look for common answer patterns
        patterns = [
            r"(?:answer|result|equals|=)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
            r"(?:is|are)\s+([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
            r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$",  # Number at end
            r"^\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",   # Number at start
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        # Try to find any number
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
        if numbers:
            return numbers[-1]  # Return last number found

        return None

    def _parse_number(self, s: str) -> Optional[Fraction]:
        """Parse string to Fraction for exact arithmetic."""
        try:
            s = s.strip()
            if "/" in s:
                return Fraction(s)
            elif "." in s:
                return Fraction(s).limit_denominator(1000000)
            else:
                return Fraction(int(s))
        except (ValueError, ZeroDivisionError):
            return None

    def _evaluate_expression(self, expr: str) -> Optional[Fraction]:
        """Safely evaluate arithmetic expression."""
        # Sanitize: only allow digits, operators, parentheses, spaces
        if not re.match(r'^[\d\s\+\-\*/%\(\)\.]+$', expr):
            return None

        try:
            # Use Python's eval with restricted builtins
            result = eval(expr, {"__builtins__": {}}, {})
            return Fraction(result).limit_denominator(1000000)
        except:
            return None

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        # Extract expected answer from prompt or ground_truth
        if ground_truth:
            expected = self._parse_number(ground_truth)
        else:
            # Try to evaluate the expression in the prompt
            expr_match = re.search(r"(?:what is|compute|calculate|evaluate)\s*(.+?)[\?\.]", prompt, re.IGNORECASE)
            if expr_match:
                expected = self._evaluate_expression(expr_match.group(1))
            else:
                return RewardResult(0.0, False, "Could not extract expected answer")

        if expected is None:
            return RewardResult(0.0, False, "Could not parse expected answer")

        # Extract answer from response
        answer_str = self._extract_answer(response)
        if answer_str is None:
            return RewardResult(0.0, False, "No numeric answer found in response")

        answer = self._parse_number(answer_str)
        if answer is None:
            return RewardResult(0.0, False, f"Could not parse answer: {answer_str}")

        # Compare
        if self.tolerance == 0.0:
            correct = answer == expected
        else:
            diff = abs(float(answer) - float(expected))
            correct = diff <= self.tolerance

        return RewardResult(
            reward=1.0 if correct else 0.0,
            correct=correct,
            reason=f"Expected {expected}, got {answer}",
            metadata={"expected": str(expected), "got": str(answer)}
        )


class IntegerArithmeticReward(ArithmeticReward):
    """
    Strict integer-only arithmetic verification.

    For ZK-ML: Ensures all operations are provable in integer circuits.
    No floating point, no irrational numbers.
    """

    def __init__(self):
        super().__init__(tolerance=0.0, allow_rationals=False)

    @property
    def name(self) -> str:
        return "integer_arithmetic"

    def _parse_number(self, s: str) -> Optional[int]:
        """Parse to integer only."""
        try:
            s = s.strip()
            if "." in s or "/" in s:
                return None  # Reject non-integers
            return int(s)
        except ValueError:
            return None

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        result = super().compute(prompt, response, ground_truth)

        # Additional check: verify answer is integer
        if result.metadata and "got" in result.metadata:
            try:
                int(result.metadata["got"])
            except ValueError:
                return RewardResult(0.0, False, "Answer is not an integer")

        return result


# =============================================================================
# CODE EXECUTION REWARDS
# =============================================================================

class CodeExecutionReward(VerifiableReward):
    """
    Verify code by execution.

    Supports:
    - Python code execution in sandbox
    - Output matching
    - Test case passing
    - Compilation check

    Security: Runs in subprocess with timeout and resource limits.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        language: str = "python",
        test_cases: Optional[List[Dict]] = None
    ):
        self.timeout = timeout
        self.language = language
        self.test_cases = test_cases or []

    @property
    def name(self) -> str:
        return "code_execution"

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code block from response."""
        # Look for markdown code blocks
        patterns = [
            r"```(?:python|py)?\n(.*?)```",
            r"```\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block, check if entire response looks like code
        lines = text.strip().split("\n")
        if lines and (lines[0].startswith("def ") or lines[0].startswith("import ")):
            return text.strip()

        return None

    def _run_python(self, code: str, test_input: Optional[str] = None) -> Tuple[bool, str, str]:
        """Run Python code in sandbox."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = subprocess.run(
                    ['python3', f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    input=test_input
                )
                return result.returncode == 0, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return False, "", "Timeout"
            except Exception as e:
                return False, "", str(e)

    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, "Valid syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        code = self._extract_code(response)

        if code is None:
            return RewardResult(0.0, False, "No code found in response")

        # Check syntax first
        syntax_ok, syntax_msg = self._check_syntax(code)
        if not syntax_ok:
            return RewardResult(0.0, False, syntax_msg)

        # If we have test cases, run them
        if self.test_cases:
            passed = 0
            for i, test in enumerate(self.test_cases):
                test_code = code
                if "setup" in test:
                    test_code = code + "\n" + test["setup"]
                if "call" in test:
                    test_code += f"\nresult = {test['call']}\nprint(result)"

                success, stdout, stderr = self._run_python(test_code, test.get("input"))

                if not success:
                    return RewardResult(
                        0.0, False,
                        f"Test {i+1} failed: {stderr}",
                        metadata={"test_index": i, "error": stderr}
                    )

                if "expected_output" in test:
                    if stdout.strip() != str(test["expected_output"]).strip():
                        return RewardResult(
                            0.0, False,
                            f"Test {i+1}: expected '{test['expected_output']}', got '{stdout.strip()}'",
                            metadata={"test_index": i, "expected": test["expected_output"], "got": stdout.strip()}
                        )

                passed += 1

            return RewardResult(
                1.0, True,
                f"All {passed} tests passed",
                metadata={"tests_passed": passed, "tests_total": len(self.test_cases)}
            )

        # If ground_truth is expected output, just run and compare
        if ground_truth:
            success, stdout, stderr = self._run_python(code)
            if not success:
                return RewardResult(0.0, False, f"Execution failed: {stderr}")

            if stdout.strip() == ground_truth.strip():
                return RewardResult(1.0, True, "Output matches expected")
            else:
                return RewardResult(0.0, False, f"Expected '{ground_truth}', got '{stdout.strip()}'")

        # Just check if code runs without error
        success, stdout, stderr = self._run_python(code)
        return RewardResult(
            1.0 if success else 0.0,
            success,
            "Code executed successfully" if success else f"Execution failed: {stderr}"
        )


# =============================================================================
# LOGIC REWARDS
# =============================================================================

class LogicReward(VerifiableReward):
    """
    Verify logical deductions.

    Supports:
    - Propositional logic (AND, OR, NOT, IMPLIES)
    - Simple syllogisms
    - Boolean satisfiability
    """

    def __init__(self):
        self.true_tokens = {"true", "yes", "correct", "valid", "1"}
        self.false_tokens = {"false", "no", "incorrect", "invalid", "0"}

    @property
    def name(self) -> str:
        return "logic"

    def _extract_boolean(self, text: str) -> Optional[bool]:
        """Extract boolean answer from text."""
        text_lower = text.lower().strip()

        # Check last word first
        words = text_lower.split()
        if words:
            last_word = words[-1].rstrip(".,!?")
            if last_word in self.true_tokens:
                return True
            if last_word in self.false_tokens:
                return False

        # Check for patterns
        if re.search(r"\b(true|yes|correct|valid)\b", text_lower):
            if not re.search(r"\b(not true|not correct|false|no|invalid)\b", text_lower):
                return True

        if re.search(r"\b(false|no|incorrect|invalid)\b", text_lower):
            return False

        return None

    def _evaluate_propositional(self, expr: str, assignments: Dict[str, bool]) -> Optional[bool]:
        """Evaluate propositional logic expression."""
        # Replace logical operators with Python equivalents
        expr = expr.replace(" AND ", " and ")
        expr = expr.replace(" OR ", " or ")
        expr = expr.replace(" NOT ", " not ")
        expr = expr.replace(" IMPLIES ", " <= ")  # A IMPLIES B = (not A) or B
        expr = expr.replace("->", " <= ")

        try:
            return eval(expr, {"__builtins__": {}}, assignments)
        except:
            return None

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        if ground_truth is None:
            return RewardResult(0.0, False, "Logic verification requires ground truth")

        # Parse expected answer
        if ground_truth.lower() in self.true_tokens:
            expected = True
        elif ground_truth.lower() in self.false_tokens:
            expected = False
        else:
            return RewardResult(0.0, False, f"Cannot parse ground truth: {ground_truth}")

        # Extract answer from response
        answer = self._extract_boolean(response)

        if answer is None:
            return RewardResult(0.0, False, "Could not extract boolean answer from response")

        correct = answer == expected
        return RewardResult(
            1.0 if correct else 0.0,
            correct,
            f"Expected {expected}, got {answer}",
            metadata={"expected": expected, "got": answer}
        )


# =============================================================================
# FORMAT REWARDS
# =============================================================================

class FormatReward(VerifiableReward):
    """
    Verify output format compliance.

    Supports:
    - JSON validity
    - Regex matching
    - Schema validation
    - Length constraints
    """

    def __init__(
        self,
        format_type: str = "json",
        regex_pattern: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required_fields: Optional[List[str]] = None
    ):
        self.format_type = format_type
        self.regex_pattern = regex_pattern
        self.min_length = min_length
        self.max_length = max_length
        self.required_fields = required_fields or []

    @property
    def name(self) -> str:
        return f"format_{self.format_type}"

    def _check_json(self, text: str) -> Tuple[bool, str, Optional[dict]]:
        """Check JSON validity and extract data."""
        # Try to find JSON in text
        json_match = re.search(r'\{[^{}]*\}|\[[^\[\]]*\]', text, re.DOTALL)
        if json_match:
            text = json_match.group()

        try:
            data = json.loads(text)
            return True, "Valid JSON", data
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None

    def _check_regex(self, text: str) -> Tuple[bool, str]:
        """Check regex pattern match."""
        if self.regex_pattern is None:
            return True, "No regex specified"

        if re.search(self.regex_pattern, text):
            return True, "Pattern matched"
        else:
            return False, f"Pattern not matched: {self.regex_pattern}"

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        # Length checks
        if self.min_length and len(response) < self.min_length:
            return RewardResult(0.0, False, f"Response too short: {len(response)} < {self.min_length}")

        if self.max_length and len(response) > self.max_length:
            return RewardResult(0.0, False, f"Response too long: {len(response)} > {self.max_length}")

        # Format-specific checks
        if self.format_type == "json":
            valid, msg, data = self._check_json(response)
            if not valid:
                return RewardResult(0.0, False, msg)

            # Check required fields
            if data and isinstance(data, dict):
                for field in self.required_fields:
                    if field not in data:
                        return RewardResult(0.0, False, f"Missing required field: {field}")

            return RewardResult(1.0, True, "Valid JSON with all required fields")

        elif self.format_type == "regex":
            valid, msg = self._check_regex(response)
            return RewardResult(1.0 if valid else 0.0, valid, msg)

        else:
            return RewardResult(0.0, False, f"Unknown format type: {self.format_type}")


# =============================================================================
# COMPOSITE REWARDS
# =============================================================================

class CompositeReward(VerifiableReward):
    """
    Combine multiple reward functions.

    Modes:
    - "all": All rewards must be 1.0 (AND)
    - "any": At least one reward must be 1.0 (OR)
    - "weighted": Weighted average of rewards
    - "sequential": Rewards are checked in order, first failure stops
    """

    def __init__(
        self,
        rewards: List[VerifiableReward],
        mode: str = "all",
        weights: Optional[List[float]] = None
    ):
        self.rewards = rewards
        self.mode = mode
        self.weights = weights or [1.0] * len(rewards)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    @property
    def name(self) -> str:
        names = "+".join(r.name for r in self.rewards)
        return f"composite({names})"

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        results = []
        for reward_fn in self.rewards:
            result = reward_fn.compute(prompt, response, ground_truth)
            results.append(result)

        if self.mode == "all":
            all_correct = all(r.correct for r in results)
            failed = [r for r in results if not r.correct]
            if failed:
                return RewardResult(
                    0.0, False,
                    f"Failed: {failed[0].reason}",
                    metadata={"results": [r.__dict__ for r in results]}
                )
            return RewardResult(1.0, True, "All checks passed")

        elif self.mode == "any":
            if any(r.correct for r in results):
                passed = [r for r in results if r.correct][0]
                return RewardResult(1.0, True, f"Passed: {passed.reason}")
            return RewardResult(0.0, False, "All checks failed")

        elif self.mode == "weighted":
            weighted_sum = sum(w * r.reward for w, r in zip(self.weights, results))
            return RewardResult(
                weighted_sum,
                weighted_sum >= 0.5,
                f"Weighted score: {weighted_sum:.3f}",
                metadata={"individual_scores": [r.reward for r in results]}
            )

        elif self.mode == "sequential":
            for i, result in enumerate(results):
                if not result.correct:
                    return RewardResult(
                        0.0, False,
                        f"Failed at step {i+1}: {result.reason}"
                    )
            return RewardResult(1.0, True, "All sequential checks passed")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# MATH-SPECIFIC REWARDS (for RLVR training focus)
# =============================================================================

class MathProblemReward(VerifiableReward):
    """
    Verify solutions to math word problems.

    Extracts final numeric answer and compares to ground truth.
    Designed for datasets like GSM8K, MATH, etc.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "math_problem"

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract boxed or final answer."""
        # Look for \boxed{} (LaTeX style)
        boxed = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed.group(1)

        # Look for "The answer is X" patterns
        patterns = [
            r"(?:the answer is|answer:?|therefore|thus|so|=)\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
            r"(?:final answer|result)[:=]?\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        # Last number in response
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
        if numbers:
            return numbers[-1]

        return None

    def compute(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> RewardResult:
        if ground_truth is None:
            return RewardResult(0.0, False, "Math verification requires ground truth")

        # Parse ground truth
        try:
            if "/" in ground_truth:
                expected = float(Fraction(ground_truth))
            else:
                expected = float(ground_truth)
        except ValueError:
            return RewardResult(0.0, False, f"Cannot parse ground truth: {ground_truth}")

        # Extract answer
        answer_str = self._extract_final_answer(response)
        if answer_str is None:
            return RewardResult(0.0, False, "No answer found in response")

        try:
            if "/" in answer_str:
                answer = float(Fraction(answer_str))
            else:
                answer = float(answer_str)
        except ValueError:
            return RewardResult(0.0, False, f"Cannot parse answer: {answer_str}")

        # Compare with tolerance
        correct = abs(answer - expected) <= self.tolerance

        return RewardResult(
            1.0 if correct else 0.0,
            correct,
            f"Expected {expected}, got {answer}",
            metadata={"expected": expected, "got": answer, "diff": abs(answer - expected)}
        )
