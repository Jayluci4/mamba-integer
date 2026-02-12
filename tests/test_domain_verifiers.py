"""
Tests for domain-specific verifier stack.

Tests each of the 5 domain verifiers + IV-GRPO reward.
Run: python -m pytest tests/test_domain_verifiers.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.rlvr.domain_verifiers import (
    SolidityVerifier,
    SQLVerifier,
    MathVerifier,
    SECFinanceVerifier,
    EnglishVerifier,
    IVGRPOReward,
    create_verifier,
    create_iv_grpo_verifier,
)


# =============================================================================
# SOLIDITY VERIFIER TESTS
# =============================================================================

class TestSolidityVerifier:
    def setup_method(self):
        self.v = SolidityVerifier()

    def test_valid_contract(self):
        response = """Here's a simple ERC20-like token:
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;

    constructor(uint256 _initialSupply) {
        balances[msg.sender] = _initialSupply;
        totalSupply = _initialSupply;
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }
}
```"""
        result = self.v.compute("Write a simple token contract", response)
        assert result.correct, f"Should compile: {result.reason}"
        assert result.reward >= 0.6

    def test_invalid_contract(self):
        response = """```solidity
pragma solidity ^0.8.0;
contract Broken {
    function x() public {
        uint256 y = ;  // syntax error
    }
}
```"""
        result = self.v.compute("Write a contract", response)
        assert not result.correct
        assert result.reward == 0.0

    def test_no_solidity_code(self):
        response = "I think you should use Hardhat for testing."
        result = self.v.compute("Write a contract", response)
        assert not result.correct
        assert "No Solidity code" in result.reason

    def test_required_functions(self):
        v = SolidityVerifier(required_functions=["transfer", "balanceOf"])
        response = """```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Token {
    mapping(address => uint256) public balances;

    function transfer(address to, uint256 amount) public returns (bool) {
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }
}
```"""
        result = v.compute("Write token with transfer and balanceOf", response)
        assert result.correct
        assert result.reward >= 0.8

    def test_missing_pragma_auto_added(self):
        response = """```solidity
contract Minimal {
    uint256 public x;
    function set(uint256 _x) public { x = _x; }
}
```"""
        result = self.v.compute("Write minimal contract", response)
        assert result.correct, f"Auto-pragma should work: {result.reason}"

    def test_compile_method(self):
        code = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract Test {
    function add(uint a, uint b) public pure returns (uint) {
        return a + b;
    }
}"""
        comp = self.v.compile(code)
        assert comp["success"]
        assert "Test" in comp["contracts"]
        assert comp["contracts"]["Test"]["gas_estimate"] > 0


# =============================================================================
# SQL VERIFIER TESTS
# =============================================================================

class TestSQLVerifier:
    def test_basic_query(self):
        schema = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO users VALUES (1, 'Alice', 30);
        INSERT INTO users VALUES (2, 'Bob', 25);
        INSERT INTO users VALUES (3, 'Charlie', 35);
        """
        v = SQLVerifier(schema=schema, setup_sql=None)
        response = """```sql
SELECT name, age FROM users WHERE age > 28;
```"""
        result = v.compute("Find users over 28", response)
        assert result.correct, f"Should execute: {result.reason}"
        assert result.metadata["row_count"] == 2

    def test_invalid_query(self):
        v = SQLVerifier()
        response = """```sql
SELECTT * FORM nonexistent;
```"""
        result = v.compute("Query something", response)
        assert not result.correct
        assert "SQL error" in result.reason or result.reward == 0.0

    def test_with_ground_truth_row_count(self):
        schema = """
        CREATE TABLE products (id INTEGER, name TEXT, price REAL);
        INSERT INTO products VALUES (1, 'Widget', 9.99);
        INSERT INTO products VALUES (2, 'Gadget', 19.99);
        INSERT INTO products VALUES (3, 'Doohickey', 4.99);
        """
        v = SQLVerifier(schema=schema)
        response = "SELECT * FROM products WHERE price < 10.0;"
        ground_truth = '{"row_count": 2}'
        result = v.compute("Find cheap products", response, ground_truth)
        assert result.correct
        assert result.reward >= 0.7

    def test_with_expected_values(self):
        schema = """
        CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT);
        INSERT INTO employees VALUES (1, 'Alice', 'Engineering');
        INSERT INTO employees VALUES (2, 'Bob', 'Marketing');
        """
        v = SQLVerifier(schema=schema)
        response = "SELECT name FROM employees WHERE dept = 'Engineering';"
        ground_truth = '{"values": ["Alice"]}'
        result = v.compute("Find engineers", response, ground_truth)
        assert result.correct

    def test_no_sql_found(self):
        v = SQLVerifier()
        result = v.compute("Write a query", "I recommend using PostgreSQL.")
        assert not result.correct
        assert "No SQL" in result.reason

    def test_aggregate_query(self):
        schema = """
        CREATE TABLE sales (id INTEGER, amount REAL, region TEXT);
        INSERT INTO sales VALUES (1, 100.0, 'North');
        INSERT INTO sales VALUES (2, 200.0, 'South');
        INSERT INTO sales VALUES (3, 150.0, 'North');
        """
        v = SQLVerifier(schema=schema)
        response = "SELECT region, SUM(amount) as total FROM sales GROUP BY region;"
        result = v.compute("Total sales by region", response)
        assert result.correct
        assert result.metadata["row_count"] == 2

    def test_expected_columns(self):
        schema = "CREATE TABLE t (a INTEGER, b TEXT, c REAL);"
        v = SQLVerifier(schema=schema, expected_columns=["a", "b"])
        response = "SELECT a, b FROM t;"
        result = v.compute("Get a and b", response)
        assert result.correct


# =============================================================================
# MATH VERIFIER TESTS
# =============================================================================

class TestMathVerifier:
    def setup_method(self):
        self.v = MathVerifier()

    def test_integer_answer(self):
        result = self.v.compute(
            "What is 7 * 8?", "The answer is 56", ground_truth="56"
        )
        assert result.correct
        assert result.reward == 1.0

    def test_fraction_answer(self):
        result = self.v.compute(
            "What is 1/3 + 1/6?", "The answer is 1/2", ground_truth="1/2"
        )
        assert result.correct

    def test_decimal_answer(self):
        result = self.v.compute(
            "What is 22/7?",
            "The answer is 3.142857",
            ground_truth="3.142857142857",
        )
        assert result.correct

    def test_wrong_answer(self):
        result = self.v.compute(
            "What is 5 + 3?", "The answer is 9", ground_truth="8"
        )
        assert not result.correct
        assert result.reward == 0.0

    def test_boxed_latex_answer(self):
        result = self.v.compute(
            "Solve x^2 = 16",
            "x = ±4, so the positive solution is \\boxed{4}",
            ground_truth="4",
        )
        assert result.correct

    def test_symbolic_equivalence(self):
        result = self.v.compute(
            "Simplify (x+1)^2",
            "The answer is x**2 + 2*x + 1.",
            ground_truth="(x+1)**2",
        )
        assert result.correct

    def test_no_answer_found(self):
        result = self.v.compute(
            "What is 2+2?", "I'm not sure about this.", ground_truth="4"
        )
        assert not result.correct

    def test_no_ground_truth(self):
        result = self.v.compute("What is pi?", "3.14159")
        assert not result.correct
        assert "requires ground truth" in result.reason


# =============================================================================
# SEC/FINANCE VERIFIER TESTS
# =============================================================================

class TestSECFinanceVerifier:
    def setup_method(self):
        self.v = SECFinanceVerifier()

    def test_good_summary(self):
        response = """
Apple Inc. (AAPL) Q4 2024 Financial Summary:

- Revenue: $89.5 billion, up 6% year-over-year
- Net income: $22.9 billion, representing a 25.6% margin
- Earnings per share: $1.46, beating consensus estimates of $1.39
- Services revenue reached $22.2 billion, a new all-time high
- Cash and equivalents: $162.1 billion

The company returned $25 billion to shareholders through dividends
and share repurchases during the quarter.
"""
        result = self.v.compute("Summarize Apple's Q4 earnings", response)
        assert result.correct, f"Should be good: {result.reason}"
        assert result.reward >= 0.6

    def test_empty_response(self):
        result = self.v.compute("Summarize earnings", "")
        assert not result.correct

    def test_missing_sections(self):
        v = SECFinanceVerifier(required_sections=["revenue", "net income", "guidance"])
        response = "Revenue was $10B. Net income was $2B."
        result = v.compute("Full earnings summary", response)
        # Should get partial credit for having 2/3 sections
        assert result.metadata["word_count"] > 0

    def test_unrealistic_figures(self):
        response = """
Revenue: $999,999,999,999,999 (that's $999 trillion)
Net income: $500 quadrillion
"""
        result = self.v.compute("Summarize earnings", response)
        # Should flag unrealistic figures
        assert result.reward < 0.8

    def test_with_ground_truth(self):
        response = "Revenue was $50.2 billion with net income of $12.1 billion."
        ground_truth = '{"revenue": "50.2 billion", "net_income": "12.1 billion"}'
        result = self.v.compute("Summarize Q3", response, ground_truth)
        assert result.correct


# =============================================================================
# ENGLISH VERIFIER TESTS
# =============================================================================

class TestEnglishVerifier:
    def setup_method(self):
        self.v = EnglishVerifier()

    def test_good_text(self):
        response = """The quick brown fox jumps over the lazy dog. This is a
well-known pangram that contains every letter of the English alphabet at
least once. It has been used since the late 19th century for testing
typewriters and computer fonts. The sentence is particularly useful
because of its brevity and completeness."""
        result = self.v.compute("Write about pangrams", response)
        assert result.correct, f"Should be good: {result.reason}"
        assert result.reward >= 0.5

    def test_empty_response(self):
        result = self.v.compute("Write something", "")
        assert not result.correct

    def test_too_short(self):
        result = self.v.compute("Write a paragraph", "Hello.")
        assert not result.correct
        assert "Too short" in result.reason

    def test_gibberish(self):
        response = "asdf jkl; qwer uiop zxcv bnm, asdf jkl; qwer uiop zxcv bnm, asdf jkl; qwer"
        result = self.v.compute("Write something", response)
        # Gibberish should score low on coherence and fluency
        assert result.reward < 0.6

    def test_repetitive_text(self):
        response = "The cat sat on the mat. " * 20
        result = self.v.compute("Write about cats", response)
        # Repetitive text should score lower on coherence
        assert result.metadata["coherence"] < 1.0

    def test_structured_response(self):
        response = """There are several key points to consider:

- First, the implementation should be efficient.
- Second, the code must be well-tested.
- Third, documentation is essential for maintenance.

In conclusion, these principles guide good software development."""
        result = self.v.compute("What makes good software?", response)
        assert result.correct


# =============================================================================
# IV-GRPO REWARD TESTS
# =============================================================================

class TestIVGRPOReward:
    def test_with_base_reward(self):
        base = MathVerifier()
        iv = IVGRPOReward(base_reward=base, alpha=0.3, beta=1.0)
        result = iv.compute("What is 2+2?", "The answer is 4", ground_truth="4")
        assert result.correct

    def test_kl_computation(self):
        import torch

        iv = IVGRPOReward(alpha=0.3, beta=1.0)

        # Identical logits → KL = 0
        logits = torch.randn(1, 10, 100)
        kl = iv.compute_kl_divergence(logits, logits)
        assert abs(kl) < 1e-5, f"Same logits should give KL≈0, got {kl}"

        # Different logits → KL > 0
        float_logits = torch.randn(1, 10, 100)
        int_logits = float_logits + torch.randn_like(float_logits) * 0.5
        kl = iv.compute_kl_divergence(float_logits, int_logits)
        assert kl > 0, f"Different logits should give KL>0, got {kl}"

    def test_reward_with_kl(self):
        import torch

        base = MathVerifier()
        iv = IVGRPOReward(base_reward=base, alpha=0.3, beta=1.0)

        float_logits = torch.randn(1, 10, 100)
        # Nearly identical → high consistency reward
        int_logits = float_logits + torch.randn_like(float_logits) * 0.01

        result = iv.compute_reward_with_kl(
            prompt="What is 2+2?",
            response="The answer is 4",
            float_logits=float_logits,
            int_logits=int_logits,
            ground_truth="4",
        )
        assert result.correct
        assert result.metadata["kl_divergence"] < 0.1
        assert result.metadata["r_consistency"] > 0.9

    def test_high_kl_penalty(self):
        import torch

        base = MathVerifier()
        iv = IVGRPOReward(base_reward=base, alpha=0.3, beta=2.0)

        float_logits = torch.randn(1, 10, 100)
        # Very different → high KL → low consistency reward
        int_logits = torch.randn(1, 10, 100) * 5

        result = iv.compute_reward_with_kl(
            prompt="What is 2+2?",
            response="The answer is 4",
            float_logits=float_logits,
            int_logits=int_logits,
            ground_truth="4",
        )
        # Base reward is correct but KL penalty should reduce total
        assert result.metadata["r_consistency"] < 0.5

    def test_factory_function(self):
        iv = create_iv_grpo_verifier("math", alpha=0.5, tolerance=1e-3)
        assert "math" in iv.name
        result = iv.compute("What is 1+1?", "2", ground_truth="2")
        assert result.correct


# =============================================================================
# FACTORY TESTS
# =============================================================================

class TestFactory:
    def test_create_all_domains(self):
        for domain in ["solidity", "sql", "math", "sec_finance", "english"]:
            v = create_verifier(domain)
            assert v.name is not None

    def test_unknown_domain(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            create_verifier("unknown_domain")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
