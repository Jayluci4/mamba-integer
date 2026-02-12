"""
Domain-Specific Verifier Stack for GRPO Post-Training

Five domain verifiers matching the post-training playbook:
1. SolidityVerifier  — solc compilation + gas analysis
2. SQLVerifier       — SQLite execution + result matching
3. MathVerifier      — SymPy symbolic + numerical checking
4. SECFinanceVerifier — format validation + entity extraction
5. EnglishVerifier   — heuristic fluency scoring (upgradeable to perplexity)

Plus: IVGRPOReward — float vs integer path KL divergence for IV-GRPO

All verifiers extend VerifiableReward with the same interface:
    compute(prompt, response, ground_truth) -> RewardResult
"""

import re
import json
import math
import sqlite3
import tempfile
import subprocess
from abc import ABC
from typing import Optional, List, Dict, Any, Tuple
from .rewards import VerifiableReward, RewardResult


# =============================================================================
# 1. SOLIDITY VERIFIER
# =============================================================================

class SolidityVerifier(VerifiableReward):
    """
    Verify Solidity smart contracts via solc compilation.

    Reward signals (composable):
    - Compiles without errors        → base reward
    - No warnings                    → bonus
    - Gas estimate below threshold   → bonus
    - Contains required functions     → bonus

    Requires: py-solc-x with solc 0.8.28+ installed
    """

    def __init__(
        self,
        solc_version: str = "0.8.28",
        check_warnings: bool = True,
        max_gas: Optional[int] = None,
        required_functions: Optional[List[str]] = None,
        required_interfaces: Optional[List[str]] = None,
    ):
        self.solc_version = solc_version
        self.check_warnings = check_warnings
        self.max_gas = max_gas
        self.required_functions = required_functions or []
        self.required_interfaces = required_interfaces or []

        # Lazy import — don't fail if solc not installed during import
        self._solcx = None

    def _get_solcx(self):
        if self._solcx is None:
            import solcx
            solcx.set_solc_version(self.solc_version)
            self._solcx = solcx
        return self._solcx

    @property
    def name(self) -> str:
        return "solidity"

    def _extract_solidity(self, text: str) -> Optional[str]:
        """Extract Solidity code from model response."""
        # Try markdown code blocks first
        patterns = [
            r"```(?:solidity|sol)\n(.*?)```",
            r"```\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if "pragma solidity" in code or "contract " in code:
                    return code

        # Try raw code detection
        if "pragma solidity" in text or "contract " in text:
            # Find the Solidity code region
            lines = text.split("\n")
            sol_lines = []
            in_code = False
            brace_depth = 0
            for line in lines:
                stripped = line.strip()
                if "pragma solidity" in stripped or (
                    "contract " in stripped and not in_code
                ):
                    in_code = True
                if in_code:
                    sol_lines.append(line)
                    brace_depth += stripped.count("{") - stripped.count("}")
                    if brace_depth <= 0 and len(sol_lines) > 1:
                        break
            if sol_lines:
                return "\n".join(sol_lines)

        return None

    def _ensure_pragma(self, code: str) -> str:
        """Ensure code has SPDX license and pragma."""
        if "SPDX-License-Identifier" not in code:
            code = "// SPDX-License-Identifier: MIT\n" + code
        if "pragma solidity" not in code:
            code = code.replace(
                "// SPDX-License-Identifier: MIT",
                "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;",
            )
        return code

    def _check_required_functions(self, abi: list) -> Tuple[bool, List[str]]:
        """Check if ABI contains required function signatures."""
        abi_functions = set()
        for item in abi:
            if item.get("type") == "function":
                abi_functions.add(item["name"])

        missing = [f for f in self.required_functions if f not in abi_functions]
        return len(missing) == 0, missing

    def _estimate_gas(self, bytecode: str) -> int:
        """Estimate deployment gas from bytecode size."""
        # Rough estimate: 200 gas per byte + 32000 base
        byte_len = len(bytecode) // 2 if bytecode.startswith("0x") else len(bytecode) // 2
        return 32000 + 200 * byte_len

    def compile(self, code: str) -> Dict[str, Any]:
        """
        Compile Solidity code and return structured result.

        Returns dict with keys:
            success: bool
            errors: list of error strings
            warnings: list of warning strings
            contracts: dict of contract_name -> {abi, bin, gas_estimate}
        """
        solcx = self._get_solcx()
        code = self._ensure_pragma(code)

        result = {
            "success": False,
            "errors": [],
            "warnings": [],
            "contracts": {},
        }

        try:
            compiled = solcx.compile_source(
                code,
                output_values=["abi", "bin", "bin-runtime"],
                solc_version=self.solc_version,
            )

            for key, contract_data in compiled.items():
                # key format: "<stdin>:ContractName"
                contract_name = key.split(":")[-1]
                bytecode = contract_data.get("bin", "")
                abi = contract_data.get("abi", [])

                result["contracts"][contract_name] = {
                    "abi": abi,
                    "bytecode": bytecode,
                    "runtime_bytecode": contract_data.get("bin-runtime", ""),
                    "gas_estimate": self._estimate_gas(bytecode),
                }

            result["success"] = True

        except Exception as e:
            error_str = str(e)
            # Parse solc error output
            for line in error_str.split("\n"):
                line = line.strip()
                if "Error:" in line:
                    result["errors"].append(line)
                elif "Warning:" in line:
                    result["warnings"].append(line)
            if not result["errors"]:
                result["errors"].append(error_str[:500])

        return result

    def compute(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> RewardResult:
        code = self._extract_solidity(response)
        if code is None:
            return RewardResult(0.0, False, "No Solidity code found in response")

        comp = self.compile(code)

        if not comp["success"]:
            return RewardResult(
                0.0,
                False,
                f"Compilation failed: {comp['errors'][0] if comp['errors'] else 'unknown error'}",
                metadata={"errors": comp["errors"]},
            )

        # Base: compiles = 0.6
        reward = 0.6
        reasons = ["Compiles successfully"]

        # No warnings bonus: +0.1
        if self.check_warnings and not comp["warnings"]:
            reward += 0.1
            reasons.append("no warnings")

        # Required functions check: +0.2
        if self.required_functions:
            for cname, cdata in comp["contracts"].items():
                ok, missing = self._check_required_functions(cdata["abi"])
                if ok:
                    reward += 0.2
                    reasons.append("all required functions present")
                    break
            else:
                reasons.append(f"missing functions: {missing}")

        # Gas check: +0.1
        if self.max_gas:
            for cname, cdata in comp["contracts"].items():
                if cdata["gas_estimate"] <= self.max_gas:
                    reward += 0.1
                    reasons.append(f"gas {cdata['gas_estimate']} <= {self.max_gas}")
                else:
                    reasons.append(
                        f"gas {cdata['gas_estimate']} > {self.max_gas}"
                    )

        # If no optional checks, compilation alone = 1.0
        if not self.required_functions and not self.max_gas and not self.check_warnings:
            reward = 1.0

        # Normalize to [0, 1]
        reward = min(reward, 1.0)

        return RewardResult(
            reward=reward,
            correct=reward >= 0.6,
            reason="; ".join(reasons),
            metadata={
                "contracts": list(comp["contracts"].keys()),
                "warnings": comp["warnings"],
                "gas_estimates": {
                    k: v["gas_estimate"] for k, v in comp["contracts"].items()
                },
            },
        )


# =============================================================================
# 2. SQL VERIFIER
# =============================================================================

class SQLVerifier(VerifiableReward):
    """
    Verify SQL queries via SQLite execution.

    Reward signals:
    - Query parses and executes     → base reward
    - Returns expected row count    → bonus
    - Returns expected values       → bonus
    - No runtime errors             → included in base

    Uses in-memory SQLite — fast and sandboxed.
    """

    def __init__(
        self,
        schema: Optional[str] = None,
        setup_sql: Optional[str] = None,
        timeout: float = 5.0,
        expected_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            schema: CREATE TABLE statements to set up before running query
            setup_sql: INSERT/setup statements to populate test data
            timeout: max execution time in seconds
            expected_columns: column names expected in result
        """
        self.schema = schema
        self.setup_sql = setup_sql
        self.timeout = timeout
        self.expected_columns = expected_columns

    @property
    def name(self) -> str:
        return "sql"

    def _extract_sql(self, text: str) -> Optional[str]:
        """Extract SQL query from model response."""
        # Try markdown code blocks
        patterns = [
            r"```(?:sql|SQL)\n(.*?)```",
            r"```\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sql = match.group(1).strip()
                if any(
                    kw in sql.upper()
                    for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "WITH"]
                ):
                    return sql

        # Try to find SQL directly in text
        sql_pattern = r"((?:SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\b[^;]*;?)"
        match = re.search(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _execute_sql(
        self, query: str, schema: Optional[str] = None, setup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute SQL in sandboxed SQLite and return results."""
        result = {
            "success": False,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": None,
        }

        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("PRAGMA journal_mode=WAL")

            cursor = conn.cursor()

            # Set up schema
            if schema:
                cursor.executescript(schema)
            if setup:
                cursor.executescript(setup)

            # Execute the query
            cursor.execute(query)

            # Fetch results
            if cursor.description:
                result["columns"] = [desc[0] for desc in cursor.description]
                result["rows"] = cursor.fetchall()
                result["row_count"] = len(result["rows"])

            result["success"] = True
            conn.close()

        except sqlite3.Error as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result

    def compute(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> RewardResult:
        sql = self._extract_sql(response)
        if sql is None:
            return RewardResult(0.0, False, "No SQL query found in response")

        exec_result = self._execute_sql(sql, self.schema, self.setup_sql)

        if not exec_result["success"]:
            return RewardResult(
                0.0,
                False,
                f"SQL error: {exec_result['error']}",
                metadata={"error": exec_result["error"]},
            )

        # Base: executes successfully = 0.5
        reward = 0.5
        reasons = ["Executes successfully"]

        # Check expected columns
        if self.expected_columns:
            actual_cols = [c.lower() for c in exec_result["columns"]]
            expected_cols = [c.lower() for c in self.expected_columns]
            if all(ec in actual_cols for ec in expected_cols):
                reward += 0.2
                reasons.append("expected columns present")
            else:
                missing = [ec for ec in expected_cols if ec not in actual_cols]
                reasons.append(f"missing columns: {missing}")

        # Check ground truth (expected result)
        if ground_truth:
            try:
                expected = json.loads(ground_truth)

                if isinstance(expected, dict):
                    # Check row count
                    if "row_count" in expected:
                        if exec_result["row_count"] == expected["row_count"]:
                            reward += 0.2
                            reasons.append(f"row count matches ({expected['row_count']})")
                        else:
                            reasons.append(
                                f"row count {exec_result['row_count']} != {expected['row_count']}"
                            )

                    # Check specific values
                    if "values" in expected:
                        actual_flat = [
                            str(v) for row in exec_result["rows"] for v in row
                        ]
                        expected_flat = [str(v) for v in expected["values"]]
                        if all(ev in actual_flat for ev in expected_flat):
                            reward += 0.3
                            reasons.append("expected values found")
                        else:
                            reasons.append("some expected values missing")

                elif isinstance(expected, list):
                    # Direct row comparison
                    actual_strs = [
                        [str(v) for v in row] for row in exec_result["rows"]
                    ]
                    expected_strs = [[str(v) for v in row] for row in expected]
                    if actual_strs == expected_strs:
                        reward += 0.5
                        reasons.append("exact result match")
                    else:
                        reasons.append("result mismatch")

            except json.JSONDecodeError:
                # ground_truth is plain text — check if it appears in results
                result_str = str(exec_result["rows"])
                if ground_truth.strip() in result_str:
                    reward += 0.3
                    reasons.append("ground truth found in results")
        else:
            # No ground truth — reward for execution + returning rows
            if exec_result["row_count"] > 0:
                reward += 0.3
                reasons.append(f"returned {exec_result['row_count']} rows")
            reward += 0.2  # Bonus for no ground truth needed

        reward = min(reward, 1.0)

        return RewardResult(
            reward=reward,
            correct=reward >= 0.5,
            reason="; ".join(reasons),
            metadata={
                "columns": exec_result["columns"],
                "row_count": exec_result["row_count"],
                "sample_rows": exec_result["rows"][:5],
            },
        )


# =============================================================================
# 3. MATH VERIFIER (SymPy-enhanced)
# =============================================================================

class MathVerifier(VerifiableReward):
    """
    Verify mathematical answers using SymPy symbolic computation.

    Upgrades over MathProblemReward:
    - Symbolic equivalence checking (x^2 + 2x + 1 == (x+1)^2)
    - Step-by-step validation (verify intermediate steps)
    - Multiple answer formats (LaTeX, plain text, fraction, decimal)
    - Expression simplification before comparison
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        check_steps: bool = False,
        allow_symbolic: bool = True,
    ):
        self.tolerance = tolerance
        self.check_steps = check_steps
        self.allow_symbolic = allow_symbolic

    @property
    def name(self) -> str:
        return "math"

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract mathematical answer from text."""
        # LaTeX boxed
        boxed = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed.group(1)

        # Common answer patterns — capture expressions, not just numbers
        # Use sentence boundary that doesn't match decimal points (period followed by digit)
        # (?:(?<!\d)\.|\Z) = period NOT preceded by digit, or end of string
        sent_end = r"(?:(?<!\d)\.\s|\Z)"
        patterns = [
            r"(?:the answer is|answer:?|therefore|thus|hence|so)\s+(.*?)" + sent_end,
            r"(?:final answer|result)[:=]?\s*(.*?)" + sent_end,
            r"(?:equals?|=)\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?(?:\s*[\+\-\*\^]+\s*[-+]?\w+(?:\.\d+)?)*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                candidate = match.group(1).strip().rstrip(".")
                # Must contain at least one digit or algebraic variable
                if re.search(r"[\dx]", candidate):
                    return candidate

        # Last number
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
        if numbers:
            return numbers[-1]

        return None

    def _parse_sympy(self, expr_str: str):
        """Parse string to SymPy expression."""
        import sympy
        try:
            # Clean LaTeX artifacts
            expr_str = expr_str.replace("\\frac", "")
            expr_str = expr_str.replace("\\cdot", "*")
            expr_str = expr_str.replace("\\times", "*")
            expr_str = expr_str.replace("\\div", "/")
            expr_str = expr_str.replace("^", "**")
            expr_str = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", expr_str)

            return sympy.sympify(expr_str)
        except (sympy.SympifyError, SyntaxError, TypeError):
            return None

    def _numeric_equal(self, a_str: str, b_str: str) -> bool:
        """Check numeric equality with tolerance."""
        try:
            from fractions import Fraction

            def to_float(s):
                s = s.strip()
                if "/" in s:
                    return float(Fraction(s))
                return float(s)

            a_val = to_float(a_str)
            b_val = to_float(b_str)
            return abs(a_val - b_val) <= self.tolerance
        except (ValueError, ZeroDivisionError):
            return False

    def _symbolic_equal(self, a_str: str, b_str: str) -> bool:
        """Check symbolic equivalence using SymPy."""
        import sympy
        a_expr = self._parse_sympy(a_str)
        b_expr = self._parse_sympy(b_str)

        if a_expr is None or b_expr is None:
            return False

        try:
            diff = sympy.simplify(a_expr - b_expr)
            return diff == 0
        except Exception:
            return False

    def _validate_steps(self, response: str) -> Tuple[bool, List[str]]:
        """Validate intermediate reasoning steps."""
        import sympy

        issues = []

        # Find equations/steps (lines with = sign)
        step_pattern = r"(?:step\s*\d+|:)\s*(.+?=.+?)(?:\n|$)"
        steps = re.findall(step_pattern, response, re.IGNORECASE)

        for step in steps:
            parts = step.split("=")
            if len(parts) == 2:
                lhs = self._parse_sympy(parts[0].strip())
                rhs = self._parse_sympy(parts[1].strip())
                if lhs is not None and rhs is not None:
                    try:
                        if sympy.simplify(lhs - rhs) != 0:
                            issues.append(f"Invalid step: {step.strip()}")
                    except Exception:
                        pass

        return len(issues) == 0, issues

    def compute(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> RewardResult:
        if ground_truth is None:
            return RewardResult(0.0, False, "Math verification requires ground truth")

        answer_str = self._extract_answer(response)
        if answer_str is None:
            return RewardResult(0.0, False, "No mathematical answer found in response")

        # Try numeric comparison first (fast path)
        if self._numeric_equal(answer_str, ground_truth):
            reward = 1.0
            reason = f"Numerically correct: {answer_str}"

            # Check steps if requested
            if self.check_steps:
                steps_ok, issues = self._validate_steps(response)
                if not steps_ok:
                    reward = 0.7
                    reason += f"; step issues: {issues}"

            return RewardResult(
                reward=reward,
                correct=True,
                reason=reason,
                metadata={"answer": answer_str, "ground_truth": ground_truth},
            )

        # Try symbolic comparison (slower)
        if self.allow_symbolic and self._symbolic_equal(answer_str, ground_truth):
            return RewardResult(
                reward=1.0,
                correct=True,
                reason=f"Symbolically equivalent: {answer_str} == {ground_truth}",
                metadata={"answer": answer_str, "ground_truth": ground_truth},
            )

        return RewardResult(
            reward=0.0,
            correct=False,
            reason=f"Expected {ground_truth}, got {answer_str}",
            metadata={"answer": answer_str, "ground_truth": ground_truth},
        )


# =============================================================================
# 4. SEC/FINANCE VERIFIER
# =============================================================================

class SECFinanceVerifier(VerifiableReward):
    """
    Verify SEC filing summaries and financial analysis.

    Reward signals:
    - Valid format (has required sections)    → base reward
    - Realistic financial figures             → bonus
    - Correct entity extraction              → bonus
    - Proper metric formatting               → bonus

    No execution needed — uses regex + heuristic validation.
    """

    # Common SEC filing metrics and their reasonable ranges
    METRIC_RANGES = {
        "revenue": (1e3, 1e12),          # $1K to $1T
        "net_income": (-1e11, 1e11),     # can be negative
        "total_assets": (1e3, 1e13),
        "total_liabilities": (1e3, 1e13),
        "eps": (-1000, 10000),           # earnings per share
        "pe_ratio": (0, 10000),
        "market_cap": (1e3, 1e13),
        "debt_to_equity": (0, 100),
    }

    def __init__(
        self,
        required_sections: Optional[List[str]] = None,
        required_metrics: Optional[List[str]] = None,
        check_numeric_ranges: bool = True,
    ):
        self.required_sections = required_sections or [
            "revenue",
            "net income",
        ]
        self.required_metrics = required_metrics or []
        self.check_numeric_ranges = check_numeric_ranges

    @property
    def name(self) -> str:
        return "sec_finance"

    def _extract_dollar_amounts(self, text: str) -> List[Tuple[str, float]]:
        """Extract dollar amounts and their context."""
        amounts = []

        # Patterns: $1.2B, $500M, $1,234,567, $1.2 billion
        patterns = [
            (r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|B)\b", 1e9),
            (r"\$\s*([\d,]+(?:\.\d+)?)\s*(million|M)\b", 1e6),
            (r"\$\s*([\d,]+(?:\.\d+)?)\s*(thousand|K)\b", 1e3),
            (r"\$\s*([\d,]+(?:\.\d+)?)\b", 1),
        ]

        for pattern, multiplier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_str = match.group(1).replace(",", "")
                try:
                    value = float(value_str) * multiplier
                    # Get surrounding context (20 chars before)
                    start = max(0, match.start() - 30)
                    context = text[start : match.start()].strip().lower()
                    amounts.append((context, value))
                except ValueError:
                    pass

        return amounts

    def _extract_percentages(self, text: str) -> List[Tuple[str, float]]:
        """Extract percentage values and context."""
        percentages = []
        for match in re.finditer(
            r"([\d,]+(?:\.\d+)?)\s*%", text
        ):
            try:
                value = float(match.group(1).replace(",", ""))
                start = max(0, match.start() - 30)
                context = text[start : match.start()].strip().lower()
                percentages.append((context, value))
            except ValueError:
                pass
        return percentages

    def _check_sections(self, text: str) -> Tuple[float, List[str]]:
        """Check if required sections/topics are covered."""
        text_lower = text.lower()
        found = []
        missing = []

        for section in self.required_sections:
            if section.lower() in text_lower:
                found.append(section)
            else:
                missing.append(section)

        if not self.required_sections:
            return 1.0, []

        score = len(found) / len(self.required_sections)
        return score, missing

    def _check_numeric_sanity(self, text: str) -> Tuple[float, List[str]]:
        """Check that financial figures are in realistic ranges."""
        issues = []
        amounts = self._extract_dollar_amounts(text)
        percentages = self._extract_percentages(text)

        for context, value in amounts:
            # Check for obviously wrong values
            if value < 0 and "loss" not in context and "deficit" not in context:
                issues.append(f"Negative amount without loss context: ${value:,.0f}")
            if value > 1e13:
                issues.append(f"Unrealistically large amount: ${value:,.0f}")

        for context, value in percentages:
            # Growth rates over 10000% are suspicious
            if value > 10000 and "basis points" not in context:
                issues.append(f"Unrealistic percentage: {value}%")

        total_checks = len(amounts) + len(percentages)
        if total_checks == 0:
            return 0.5, ["No financial figures found"]

        score = 1.0 - (len(issues) / max(total_checks, 1))
        return max(score, 0.0), issues

    def compute(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> RewardResult:
        if not response.strip():
            return RewardResult(0.0, False, "Empty response")

        reward = 0.0
        reasons = []

        # Section coverage (0.4 weight)
        section_score, missing = self._check_sections(response)
        reward += 0.4 * section_score
        if section_score == 1.0:
            reasons.append("all required sections present")
        elif missing:
            reasons.append(f"missing sections: {missing}")

        # Financial figure sanity (0.3 weight)
        if self.check_numeric_ranges:
            numeric_score, issues = self._check_numeric_sanity(response)
            reward += 0.3 * numeric_score
            if numeric_score >= 0.8:
                reasons.append("financial figures realistic")
            elif issues:
                reasons.append(f"numeric issues: {issues[:2]}")

        # Response quality heuristics (0.3 weight)
        quality_score = 0.0

        # Has some structure (bullet points, numbered lists, headers)
        if re.search(r"(?:^|\n)\s*[-•*]\s", response) or re.search(
            r"(?:^|\n)\s*\d+[\.)]\s", response
        ):
            quality_score += 0.5

        # Reasonable length (not too short, not too long)
        word_count = len(response.split())
        if 50 <= word_count <= 2000:
            quality_score += 0.5
        elif 20 <= word_count < 50:
            quality_score += 0.25

        reward += 0.3 * quality_score
        if quality_score >= 0.5:
            reasons.append(f"good structure ({word_count} words)")

        # Ground truth comparison
        if ground_truth:
            try:
                expected = json.loads(ground_truth)
                if isinstance(expected, dict):
                    # Check specific expected values
                    text_lower = response.lower()
                    matches = sum(
                        1 for v in expected.values() if str(v).lower() in text_lower
                    )
                    if matches > 0:
                        gt_score = matches / len(expected)
                        reward = reward * 0.5 + gt_score * 0.5
                        reasons.append(
                            f"matched {matches}/{len(expected)} expected values"
                        )
            except (json.JSONDecodeError, TypeError):
                if ground_truth.lower() in response.lower():
                    reward = min(reward + 0.2, 1.0)
                    reasons.append("ground truth found in response")

        reward = min(reward, 1.0)

        return RewardResult(
            reward=reward,
            correct=reward >= 0.5,
            reason="; ".join(reasons) if reasons else "Evaluated",
            metadata={
                "dollar_amounts": len(self._extract_dollar_amounts(response)),
                "percentages": len(self._extract_percentages(response)),
                "word_count": len(response.split()),
            },
        )


# =============================================================================
# 5. ENGLISH VERIFIER
# =============================================================================

class EnglishVerifier(VerifiableReward):
    """
    Verify English text quality using heuristic scoring.

    Reward signals (all heuristic — no learned model needed):
    - Coherence: sentence connectivity, no random tokens
    - Fluency: word-level perplexity proxy, vocabulary diversity
    - Format: proper capitalization, punctuation, paragraph structure
    - Length: appropriate response length for the prompt

    Designed to be upgraded to a model-based perplexity scorer once
    a teacher model is available.
    """

    # Common English words for vocabulary scoring
    COMMON_WORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
    }

    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 2000,
        check_grammar: bool = True,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.check_grammar = check_grammar

    @property
    def name(self) -> str:
        return "english"

    # Common English bigrams — real words almost always contain these
    _COMMON_BIGRAMS = {
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
        "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
        "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
        "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
        "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur",
    }

    def _is_real_word(self, w: str) -> bool:
        """Check if a word looks like real English (not keyboard gibberish)."""
        if w in self.COMMON_WORDS:
            return True
        if not re.match(r"^[a-z]+(?:'[a-z]+)?$", w):
            return False
        # Must have a vowel
        if not set("aeiou") & set(w):
            return False
        # Short words (<=2 chars) with vowel are OK (a, I, an, etc.)
        if len(w) <= 2:
            return True
        # For words >= 3 chars, require that >= 40% of bigrams are common
        # This catches keyboard gibberish like "asdf" (1/3=33%) and "qwer" (1/3=33%)
        # while keeping real words like "the" (2/2=100%), "answer" (3/5=60%)
        n_bigrams = len(w) - 1
        common_count = sum(
            1 for i in range(n_bigrams) if w[i:i+2] in self._COMMON_BIGRAMS
        )
        return common_count / n_bigrams >= 0.4

    def _score_coherence(self, text: str) -> float:
        """Score text coherence (0-1)."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        score = 0.0
        checks = 0

        # Check: sentences have reasonable length
        for sent in sentences:
            words = sent.split()
            checks += 1
            if 3 <= len(words) <= 50:
                score += 1.0
            elif len(words) > 0:
                score += 0.3

        # Check: no excessive repetition
        if len(sentences) >= 3:
            unique_sents = set(s.lower() for s in sentences)
            checks += 1
            repetition_ratio = len(unique_sents) / len(sentences)
            score += repetition_ratio

        # Check: no gibberish (ratio of dictionary-like words)
        words = text.lower().split()
        if words:
            real_word_count = sum(1 for w in words if self._is_real_word(w))
            checks += 1
            score += real_word_count / len(words)

        return score / checks if checks > 0 else 0.0

    def _score_fluency(self, text: str) -> float:
        """Score fluency via vocabulary diversity and common word ratio."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0.0

        score = 0.0
        checks = 0

        # Vocabulary diversity (type-token ratio)
        checks += 1
        if len(words) > 0:
            ttr = len(set(words)) / len(words)
            # Good TTR is 0.4-0.8 for natural text
            if 0.3 <= ttr <= 0.85:
                score += 1.0
            elif 0.15 <= ttr < 0.3 or 0.85 < ttr <= 0.95:
                score += 0.5
            else:
                score += 0.2

        # Common word ratio (natural text has ~50-70% common words)
        checks += 1
        common_count = sum(1 for w in words if w in self.COMMON_WORDS)
        common_ratio = common_count / len(words)
        if 0.3 <= common_ratio <= 0.75:
            score += 1.0
        elif 0.15 <= common_ratio < 0.3 or 0.75 < common_ratio <= 0.9:
            score += 0.5
        else:
            score += 0.2

        # Average word length (natural English: 4-6 chars)
        checks += 1
        avg_len = sum(len(w) for w in words) / len(words)
        if 3.5 <= avg_len <= 7.0:
            score += 1.0
        elif 2.5 <= avg_len < 3.5 or 7.0 < avg_len <= 9.0:
            score += 0.5
        else:
            score += 0.2

        return score / checks

    def _score_format(self, text: str) -> float:
        """Score formatting quality."""
        score = 0.0
        checks = 0

        # Starts with capital letter
        checks += 1
        if text and text[0].isupper():
            score += 1.0

        # Ends with punctuation
        checks += 1
        if text and text.rstrip()[-1:] in ".!?\"'":
            score += 1.0

        # Reasonable paragraph structure
        checks += 1
        lines = text.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if len(non_empty) >= 1:
            score += 1.0

        # No excessive special characters
        checks += 1
        special_ratio = sum(1 for c in text if not c.isalnum() and c not in " \n\t.,;:!?'-\"()") / max(len(text), 1)
        if special_ratio < 0.1:
            score += 1.0
        elif special_ratio < 0.2:
            score += 0.5

        return score / checks if checks > 0 else 0.0

    def compute(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> RewardResult:
        if not response.strip():
            return RewardResult(0.0, False, "Empty response")

        words = response.split()
        word_count = len(words)

        # Length check
        if word_count < self.min_words:
            return RewardResult(
                0.1, False, f"Too short: {word_count} words < {self.min_words}"
            )
        if word_count > self.max_words:
            return RewardResult(
                0.3, False, f"Too long: {word_count} words > {self.max_words}"
            )

        # Score components
        coherence = self._score_coherence(response)
        fluency = self._score_fluency(response)
        formatting = self._score_format(response)

        # Weighted combination
        reward = 0.4 * coherence + 0.35 * fluency + 0.25 * formatting

        reasons = [
            f"coherence={coherence:.2f}",
            f"fluency={fluency:.2f}",
            f"format={formatting:.2f}",
            f"words={word_count}",
        ]

        return RewardResult(
            reward=reward,
            correct=reward >= 0.5,
            reason="; ".join(reasons),
            metadata={
                "coherence": coherence,
                "fluency": fluency,
                "formatting": formatting,
                "word_count": word_count,
            },
        )


# =============================================================================
# IV-GRPO: INTEGER-VERIFIED REWARD
# =============================================================================

class IVGRPOReward(VerifiableReward):
    """
    Integer-Verified GRPO reward: measures KL divergence between
    float training path and integer-only inference path.

    R = R_correctness + α * exp(-β * KL(P_float || P_integer))

    Low KL = high reward. Trains the model toward parameter regions
    where integer quantization causes minimal distributional shift.

    This is the novel differentiator from the playbook — no existing work
    trains for quantization robustness through RL rewards.
    """

    def __init__(
        self,
        base_reward: Optional[VerifiableReward] = None,
        alpha: float = 0.3,
        beta: float = 1.0,
        max_kl: float = 10.0,
    ):
        """
        Args:
            base_reward: underlying domain verifier for R_correctness
            alpha: weight of the integer-consistency reward
            beta: KL sensitivity (higher = more penalty for divergence)
            max_kl: clamp KL divergence to prevent numerical issues
        """
        self.base_reward = base_reward
        self.alpha = alpha
        self.beta = beta
        self.max_kl = max_kl

    @property
    def name(self) -> str:
        base = self.base_reward.name if self.base_reward else "none"
        return f"iv_grpo({base})"

    def compute_kl_divergence(
        self,
        float_logits,  # torch.Tensor [batch, seq, vocab]
        int_logits,    # torch.Tensor [batch, seq, vocab]
        temperature: float = 1.0,
    ) -> float:
        """
        Compute KL(P_float || P_integer) between float and integer paths.

        Both inputs are raw logits. We convert to log-probs, then compute
        KL divergence averaged over sequence positions.
        """
        import torch
        import torch.nn.functional as F

        # Apply temperature
        float_logits = float_logits / temperature
        int_logits = int_logits / temperature

        # Convert to log-probabilities
        float_log_probs = F.log_softmax(float_logits, dim=-1)
        int_log_probs = F.log_softmax(int_logits, dim=-1)

        # KL(P_float || P_integer) = sum P_float * (log P_float - log P_integer)
        float_probs = float_log_probs.exp()
        kl = (float_probs * (float_log_probs - int_log_probs)).sum(dim=-1)

        # Average over batch and sequence
        kl_mean = kl.mean().item()

        # Clamp for numerical stability
        kl_mean = min(kl_mean, self.max_kl)
        kl_mean = max(kl_mean, 0.0)

        return kl_mean

    def compute_reward_with_kl(
        self,
        prompt: str,
        response: str,
        float_logits,
        int_logits,
        ground_truth: Optional[str] = None,
        temperature: float = 1.0,
    ) -> RewardResult:
        """
        Compute combined reward: R_correctness + α * exp(-β * KL).

        This is the main entry point during GRPO training where we have
        access to both logit paths.
        """
        # Get base correctness reward
        if self.base_reward:
            base_result = self.base_reward.compute(prompt, response, ground_truth)
            r_correctness = base_result.reward
        else:
            r_correctness = 0.0
            base_result = RewardResult(0.0, False, "No base reward")

        # Compute KL divergence
        kl = self.compute_kl_divergence(float_logits, int_logits, temperature)

        # Integer-consistency reward: exp(-β * KL)
        r_consistency = math.exp(-self.beta * kl)

        # Combined reward
        total_reward = r_correctness + self.alpha * r_consistency

        # Normalize to [0, 1+alpha] → [0, 1]
        normalized_reward = total_reward / (1.0 + self.alpha)

        return RewardResult(
            reward=normalized_reward,
            correct=base_result.correct and kl < 1.0,
            reason=f"correctness={r_correctness:.3f}, kl={kl:.4f}, consistency={r_consistency:.3f}",
            metadata={
                "r_correctness": r_correctness,
                "r_consistency": r_consistency,
                "kl_divergence": kl,
                "total_reward": total_reward,
                "base_correct": base_result.correct,
                "base_reason": base_result.reason,
            },
        )

    def compute(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> RewardResult:
        """Fallback without logits — just returns base reward."""
        if self.base_reward:
            return self.base_reward.compute(prompt, response, ground_truth)
        return RewardResult(0.0, False, "IV-GRPO requires logits for KL computation")


# =============================================================================
# DOMAIN VERIFIER REGISTRY
# =============================================================================

def create_verifier(domain: str, **kwargs) -> VerifiableReward:
    """Factory function for creating domain-specific verifiers."""
    registry = {
        "solidity": SolidityVerifier,
        "sql": SQLVerifier,
        "math": MathVerifier,
        "sec_finance": SECFinanceVerifier,
        "english": EnglishVerifier,
    }

    if domain not in registry:
        raise ValueError(
            f"Unknown domain: {domain}. Available: {list(registry.keys())}"
        )

    return registry[domain](**kwargs)


def create_iv_grpo_verifier(
    domain: str, alpha: float = 0.3, beta: float = 1.0, **domain_kwargs
) -> IVGRPOReward:
    """Create an IV-GRPO wrapped domain verifier."""
    base = create_verifier(domain, **domain_kwargs)
    return IVGRPOReward(base_reward=base, alpha=alpha, beta=beta)
