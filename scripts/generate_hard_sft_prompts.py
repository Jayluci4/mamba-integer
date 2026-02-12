#!/usr/bin/env python3
"""
Generate hard, multi-step SFT prompts across all 5 verifier domains.

Each prompt requires 3+ reasoning steps, going beyond what pretraining covers.
Pretraining teaches domain knowledge (raw text); SFT teaches structured reasoning
and verifiable output formats.

Usage:
    python scripts/generate_hard_sft_prompts.py --output data/sft_prompts_hard.jsonl
"""

import argparse
import json
import random
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Optional

random.seed(42)


# =============================================================================
# MATH PROMPTS — Multi-step problems with computed ground truths
# =============================================================================

def generate_math_prompts() -> List[Dict]:
    """Generate 250 multi-step math prompts with computed ground truths."""
    prompts = []

    # --- Category 1: Systems of equations (30 prompts) ---
    for i in range(30):
        # ax + by = c, dx + ey = f — solvable with integer solutions
        x_true = random.randint(-15, 15)
        y_true = random.randint(-15, 15)
        a, b = random.randint(1, 10), random.randint(1, 10)
        d, e = random.randint(1, 10), random.randint(1, 10)
        # Ensure non-degenerate (det != 0)
        while a * e - b * d == 0:
            e = random.randint(1, 10)
        c = a * x_true + b * y_true
        f = d * x_true + e * y_true
        prompts.append({
            "prompt": f"Solve the system of equations:\n{a}x + {b}y = {c}\n{d}x + {e}y = {f}\nFind the value of x + y.",
            "domain": "math",
            "ground_truth": str(x_true + y_true)
        })

    # --- Category 2: Combinatorics & probability (30 prompts) ---
    for i in range(15):
        n = random.randint(5, 12)
        r = random.randint(2, min(n - 1, 5))
        # C(n,r) = n! / (r! * (n-r)!)
        ans = math.comb(n, r)
        prompts.append({
            "prompt": f"A committee of {r} people must be chosen from a group of {n} candidates. How many different committees are possible?",
            "domain": "math",
            "ground_truth": str(ans)
        })

    for i in range(15):
        # Probability with multiple events
        red = random.randint(3, 8)
        blue = random.randint(3, 8)
        total = red + blue
        # P(2 red without replacement) = C(red,2) / C(total,2)
        numerator = math.comb(red, 2)
        denominator = math.comb(total, 2)
        g = math.gcd(numerator, denominator)
        num_s, den_s = numerator // g, denominator // g
        if den_s == 1:
            gt = str(num_s)
        else:
            gt = f"{num_s}/{den_s}"
        prompts.append({
            "prompt": f"A bag contains {red} red balls and {blue} blue balls. Two balls are drawn without replacement. What is the probability that both are red? Express as a simplified fraction.",
            "domain": "math",
            "ground_truth": gt
        })

    # --- Category 3: Number theory (25 prompts) ---
    for i in range(10):
        a = random.randint(50, 500)
        b = random.randint(50, 500)
        while b == a:
            b = random.randint(50, 500)
        g = math.gcd(a, b)
        l = (a * b) // g
        prompts.append({
            "prompt": f"Find the least common multiple (LCM) of {a} and {b}.",
            "domain": "math",
            "ground_truth": str(l)
        })

    for i in range(15):
        # Modular arithmetic: find x such that ax ≡ b (mod m)
        m = random.choice([7, 11, 13, 17, 19, 23])
        a = random.randint(2, m - 1)
        while math.gcd(a, m) != 1:
            a = random.randint(2, m - 1)
        x_true = random.randint(1, m - 1)
        b = (a * x_true) % m
        prompts.append({
            "prompt": f"Find the smallest positive integer x such that {a}x ≡ {b} (mod {m}).",
            "domain": "math",
            "ground_truth": str(x_true)
        })

    # --- Category 4: Geometry (25 prompts) ---
    for i in range(10):
        # Triangle area with coordinates
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)
        area_2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        # Ensure non-degenerate
        while area_2 == 0:
            x3, y3 = random.randint(-10, 10), random.randint(-10, 10)
            area_2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if area_2 % 2 == 0:
            gt = str(area_2 // 2)
        else:
            gt = f"{area_2}/2"
        prompts.append({
            "prompt": f"Find the area of the triangle with vertices at ({x1},{y1}), ({x2},{y2}), and ({x3},{y3}).",
            "domain": "math",
            "ground_truth": gt
        })

    for i in range(8):
        # Circle and chord
        r = random.randint(5, 20)
        d = random.randint(1, r - 1)  # distance from center to chord
        # chord_length = 2 * sqrt(r^2 - d^2)
        val = r * r - d * d
        # Only keep if perfect square for clean answer
        sq = int(math.isqrt(val))
        if sq * sq == val:
            gt = str(2 * sq)
        else:
            # Express as 2*sqrt(val)
            # Simplify sqrt: factor out perfect squares
            outer = 1
            inner = val
            for p in [4, 9, 16, 25, 36, 49, 64, 81, 100]:
                while inner % p == 0:
                    inner //= p
                    outer *= int(math.isqrt(p))
            gt = f"{2 * outer}" if inner == 1 else str(round(2 * math.sqrt(val), 4))
        prompts.append({
            "prompt": f"A chord is drawn in a circle of radius {r}. The distance from the center to the chord is {d}. What is the length of the chord?",
            "domain": "math",
            "ground_truth": gt
        })

    for i in range(7):
        # Sector area
        r = random.randint(3, 15)
        angle = random.choice([30, 45, 60, 90, 120, 150, 180, 270])
        # Area = (angle/360) * pi * r^2 — use pi=3.14159
        area = round((angle / 360) * 3.14159 * r * r, 2)
        prompts.append({
            "prompt": f"Find the area of a sector with radius {r} and central angle {angle} degrees. Use pi = 3.14159 and round to 2 decimal places.",
            "domain": "math",
            "ground_truth": str(area)
        })

    # --- Category 5: Word problems requiring multiple steps (40 prompts) ---
    for i in range(10):
        # Rate problems: A and B work together
        a_hours = random.randint(3, 12)
        b_hours = random.randint(3, 12)
        while b_hours == a_hours:
            b_hours = random.randint(3, 12)
        # Combined rate: 1/a + 1/b = (a+b)/(a*b), time = a*b/(a+b)
        f = Fraction(a_hours * b_hours, a_hours + b_hours)
        if f.denominator == 1:
            gt = str(f.numerator)
        else:
            gt = str(round(float(f), 4))
        prompts.append({
            "prompt": f"Worker A can complete a job in {a_hours} hours. Worker B can complete the same job in {b_hours} hours. If they work together, how many hours will it take? Round to 4 decimal places if not an integer.",
            "domain": "math",
            "ground_truth": gt
        })

    for i in range(10):
        # Mixture problems
        vol1 = random.randint(10, 50)
        pct1 = random.randint(10, 40)
        vol2 = random.randint(10, 50)
        pct2 = random.randint(50, 90)
        # Result: (vol1*pct1 + vol2*pct2) / (vol1 + vol2)
        result_pct = round((vol1 * pct1 + vol2 * pct2) / (vol1 + vol2), 2)
        prompts.append({
            "prompt": f"You mix {vol1} liters of a {pct1}% acid solution with {vol2} liters of a {pct2}% acid solution. What is the concentration of the resulting mixture? Round to 2 decimal places.",
            "domain": "math",
            "ground_truth": str(result_pct)
        })

    for i in range(10):
        # Distance-rate-time with multiple legs
        d1 = random.randint(50, 200)
        s1 = random.randint(30, 80)
        d2 = random.randint(50, 200)
        s2 = random.randint(30, 80)
        while s2 == s1:
            s2 = random.randint(30, 80)
        total_time = round(d1 / s1 + d2 / s2, 2)
        avg_speed = round((d1 + d2) / total_time, 2)
        prompts.append({
            "prompt": f"A car travels {d1} km at {s1} km/h, then {d2} km at {s2} km/h. What is the average speed for the entire trip? Round to 2 decimal places.",
            "domain": "math",
            "ground_truth": str(avg_speed)
        })

    for i in range(10):
        # Compound interest
        principal = random.choice([1000, 2000, 5000, 10000])
        rate = random.choice([5, 6, 7, 8, 10, 12])
        years = random.randint(2, 5)
        n = random.choice([1, 2, 4, 12])  # compounding frequency
        amount = round(principal * (1 + rate / 100 / n) ** (n * years), 2)
        freq_name = {1: "annually", 2: "semi-annually", 4: "quarterly", 12: "monthly"}[n]
        prompts.append({
            "prompt": f"${principal} is invested at {rate}% annual interest, compounded {freq_name}, for {years} years. What is the final amount? Round to 2 decimal places.",
            "domain": "math",
            "ground_truth": str(amount)
        })

    # --- Category 6: Sequences & series (25 prompts) ---
    for i in range(10):
        # Arithmetic series
        a1 = random.randint(1, 20)
        d = random.randint(1, 10)
        n = random.randint(10, 30)
        s = n * (2 * a1 + (n - 1) * d) // 2
        prompts.append({
            "prompt": f"Find the sum of the first {n} terms of an arithmetic sequence with first term {a1} and common difference {d}.",
            "domain": "math",
            "ground_truth": str(s)
        })

    for i in range(10):
        # Geometric series
        a1 = random.randint(1, 5)
        r = random.choice([2, 3])
        n = random.randint(5, 10)
        s = a1 * (r ** n - 1) // (r - 1)
        prompts.append({
            "prompt": f"Find the sum of the first {n} terms of a geometric sequence with first term {a1} and common ratio {r}.",
            "domain": "math",
            "ground_truth": str(s)
        })

    for i in range(5):
        # Find the nth term of a quadratic sequence
        a, b, c = random.randint(1, 5), random.randint(-5, 10), random.randint(-10, 10)
        terms = [a * k * k + b * k + c for k in range(1, 6)]
        n = random.randint(10, 20)
        ans = a * n * n + b * n + c
        terms_str = ", ".join(str(t) for t in terms)
        prompts.append({
            "prompt": f"The sequence begins: {terms_str}, ... Find the {n}th term.",
            "domain": "math",
            "ground_truth": str(ans)
        })

    # --- Category 7: Polynomial & equation solving (25 prompts) ---
    for i in range(10):
        # Quadratic with integer roots
        r1 = random.randint(-10, 10)
        r2 = random.randint(-10, 10)
        # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
        b = -(r1 + r2)
        c = r1 * r2
        prompts.append({
            "prompt": f"Find the sum of the roots of x^2 {'+' if b >= 0 else '-'} {abs(b)}x {'+' if c >= 0 else '-'} {abs(c)} = 0.",
            "domain": "math",
            "ground_truth": str(r1 + r2)
        })

    for i in range(8):
        # Product of roots
        r1 = random.randint(-10, 10)
        r2 = random.randint(-10, 10)
        b = -(r1 + r2)
        c = r1 * r2
        prompts.append({
            "prompt": f"Find the product of the roots of x^2 {'+' if b >= 0 else '-'} {abs(b)}x {'+' if c >= 0 else '-'} {abs(c)} = 0.",
            "domain": "math",
            "ground_truth": str(r1 * r2)
        })

    for i in range(7):
        # Absolute value equations: |ax + b| = c
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(1, 20)
        # Solutions: x = (c - b)/a and x = (-c - b)/a
        x1 = Fraction(c - b, a)
        x2 = Fraction(-c - b, a)
        ans = x1 + x2  # sum of solutions
        if ans.denominator == 1:
            gt = str(ans.numerator)
        else:
            gt = f"{ans.numerator}/{ans.denominator}"
        prompts.append({
            "prompt": f"Find the sum of all solutions to |{a}x + {b}| = {c}.",
            "domain": "math",
            "ground_truth": gt
        })

    # --- Category 8: Counting & logic (20 prompts) ---
    for i in range(10):
        # Digit counting
        n = random.randint(100, 999)
        # How many integers from 1 to n contain digit d
        d = random.randint(1, 9)
        count = sum(1 for x in range(1, n + 1) if str(d) in str(x))
        prompts.append({
            "prompt": f"How many integers from 1 to {n} contain the digit {d} at least once?",
            "domain": "math",
            "ground_truth": str(count)
        })

    for i in range(10):
        # Permutations with constraints
        n = random.randint(4, 7)
        # Derangements: D(n) = n! * sum((-1)^k / k!, k=0..n)
        def derangement(n):
            if n == 0: return 1
            if n == 1: return 0
            return (n - 1) * (derangement(n - 1) + derangement(n - 2))
        ans = derangement(n)
        prompts.append({
            "prompt": f"{n} people each put their hat into a pile. The hats are randomly redistributed. In how many ways can the hats be distributed so that nobody gets their own hat back? (This is the number of derangements of {n}.)",
            "domain": "math",
            "ground_truth": str(ans)
        })

    # --- Category 9: Matrix & linear algebra basics (20 prompts) ---
    for i in range(10):
        # 2x2 determinant
        a, b, c, d = [random.randint(-10, 10) for _ in range(4)]
        det = a * d - b * c
        prompts.append({
            "prompt": f"Find the determinant of the 2x2 matrix [[{a}, {b}], [{c}, {d}]].",
            "domain": "math",
            "ground_truth": str(det)
        })

    for i in range(10):
        # 3x3 determinant
        m = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        mat_str = f"[[{m[0][0]}, {m[0][1]}, {m[0][2]}], [{m[1][0]}, {m[1][1]}, {m[1][2]}], [{m[2][0]}, {m[2][1]}, {m[2][2]}]]"
        prompts.append({
            "prompt": f"Find the determinant of the 3x3 matrix {mat_str}.",
            "domain": "math",
            "ground_truth": str(det)
        })

    assert len(prompts) >= 240, f"Expected 250 math prompts, got {len(prompts)}"
    return prompts[:250]


# =============================================================================
# SQL PROMPTS — Complex queries against the 12-table default schema
# =============================================================================

# The default schema in sft_generator.py has these tables:
# employees (id, first_name, last_name, name, email, department, salary, hire_date, manager_id, age, city, title)
# customers (id, first_name, last_name, name, email, city, country, phone, signup_date, total_spent)
# orders (id, customer_id, product_id, amount, quantity, order_date, status, total)
# products (id, name, category, price, stock, description)
# departments (id, name, budget, location, manager_id)
# sales (id, employee_id, product_id, amount, sale_date, region)
# students (id, name, grade, gpa, enrollment_date)
# transactions (id, account_id, amount, type, transaction_date, description)
# inventory (id, product_id, warehouse, quantity, last_updated)
# users (id, username, email, created_at, is_active)
# reviews (id, product_id, user_id, rating, review_text, created_at)
# payments (id, order_id, amount, method, payment_date, status)

def generate_sql_prompts() -> List[Dict]:
    """Generate 200 complex SQL prompts requiring multi-step reasoning."""
    prompts = []

    # All use the default schema (no per-prompt schema override needed)

    # --- Category 1: Window functions (30 prompts) ---
    window_prompts = [
        "Rank all employees by salary within their department. Show employee name, department, salary, and rank.",
        "For each employee, show their salary and the running total of salaries ordered by hire_date.",
        "Find each employee's salary as a percentage of their department's total salary. Show name, department, salary, and percentage rounded to 1 decimal.",
        "Show each order with its amount and the previous order's amount (by order_date). Use LAG.",
        "For each department, find the employee with the highest salary using ROW_NUMBER. Show only the top earner per department.",
        "Calculate the cumulative count of orders per status, ordered by order_date.",
        "Show each sale amount alongside the department average sale amount using a window function.",
        "For each product, show the review rating and the average rating for that product as a window aggregate.",
        "Rank customers by total_spent and show their percentile rank (PERCENT_RANK).",
        "For each employee, show the difference between their salary and the next-highest salary in their department using LEAD.",
        "Find the median salary per department using PERCENTILE_CONT or a ROW_NUMBER-based approach.",
        "Show each transaction with a 3-row moving average of amounts, ordered by transaction_date.",
        "For each order, show the order_date and the number of days since the previous order by the same customer.",
        "Partition sales by region and rank them by amount. Show only the top 2 sales per region.",
        "For each product in inventory, show the quantity and the ratio of its quantity to the total across all warehouses.",
        "Show each student's GPA alongside the class average GPA and the deviation from average.",
        "Use NTILE(4) to divide employees into salary quartiles. Show name, salary, and quartile.",
        "For each payment, show the cumulative sum of payments per payment method, ordered by payment_date.",
        "Find the first and last order date per customer using FIRST_VALUE and LAST_VALUE window functions.",
        "Show each sale with its amount and the maximum sale amount in the same region as a window aggregate.",
        "For each employee, calculate their salary's Z-score within their department (how many std devs from dept mean).",
        "Rank products by the number of reviews they have. Handle ties with DENSE_RANK.",
        "Show each order's total and what percentage it represents of all orders for that customer.",
        "For each transaction, show the running balance (cumulative sum) per account_id ordered by transaction_date.",
        "Find employees whose salary is above the average salary of all employees hired in the same year.",
        "Use ROW_NUMBER to deduplicate: if multiple employees share the same name, keep only the one with the highest salary.",
        "For each department, show the salary gap between the highest and second-highest paid employee.",
        "Show each review's rating alongside the average rating for that product, and flag reviews that are more than 1 point above or below average.",
        "For each customer, show their orders with a column indicating if it's their first, second, third, etc. order.",
        "Calculate the month-over-month growth rate of order totals, grouped by month of order_date.",
    ]
    for p in window_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    # --- Category 2: CTEs and subqueries (30 prompts) ---
    cte_prompts = [
        "Using a CTE, find all departments where the average salary exceeds the company-wide average salary.",
        "Write a recursive CTE to build an employee hierarchy starting from employees with no manager (manager_id IS NULL).",
        "Using a CTE, find customers who have placed more orders than the average number of orders per customer.",
        "With a CTE, calculate each department's salary budget utilization: total salaries / department budget * 100.",
        "Using nested CTEs, first find the top 3 selling products (by total sales amount), then find all reviews for those products.",
        "Write a CTE that computes monthly revenue from orders, then find the month with the highest revenue.",
        "Using a CTE, find products that have inventory in all warehouses.",
        "Write a CTE to find customers whose total_spent exceeds the median total_spent of all customers.",
        "Using a CTE, find employees who earn more than their manager.",
        "Write a CTE that creates a date series and LEFT JOINs with orders to show days with zero orders.",
        "Using two CTEs, find the correlation between product price and average review rating.",
        "Write a CTE to identify 'loyal customers': those with 3+ orders and total_spent > $500.",
        "Using a CTE, find the department with the most diverse city representation among its employees.",
        "Write a correlated subquery to find products where every review has a rating >= 4.",
        "Using a CTE, compute the Pareto ratio: what percentage of customers account for 80% of total revenue.",
        "Write a CTE to find all employees who were hired within 30 days of another employee in the same department.",
        "Using a CTE, find the busiest order day (most orders in a single day).",
        "Write a CTE that calculates each customer's recency (days since last order) and frequency (total orders).",
        "Using a CTE, find products with declining sales (each sale amount lower than the previous one).",
        "Write a CTE to find the average time between orders for each customer.",
        "Using a CTE, identify departments where all employees are above the company-wide median salary.",
        "Write a CTE to segment customers into 'new' (1 order), 'returning' (2-3), and 'loyal' (4+) based on order count.",
        "Using a CTE, find employees whose salary rank improved (lower rank = higher salary) compared to the previous hire year.",
        "Write a CTE to find the product category with the highest average review rating.",
        "Using a CTE, calculate the retention rate: percentage of customers who placed a second order.",
        "Write a CTE to find pairs of employees in the same department with salaries within $5000 of each other.",
        "Using a CTE, find months where order count exceeded the 3-month moving average.",
        "Write a CTE to compute the coefficient of variation (stddev/mean) of salaries per department.",
        "Using a CTE, find customers who ordered the same product more than once.",
        "Write a CTE to rank departments by average employee tenure (based on hire_date).",
    ]
    for p in cte_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    # --- Category 3: Multi-table JOINs with business logic (30 prompts) ---
    join_prompts = [
        "Find the total revenue generated by each employee through the sales they made. Join employees and sales, group by employee, and show name and total revenue ordered by revenue descending.",
        "Show each order with the customer name, product name, and payment status. Join orders, customers, products, and payments.",
        "Find products that have been ordered but never reviewed. Use a LEFT JOIN between products, orders, and reviews.",
        "For each department, show the total salary, the department budget, and the remaining budget (budget - total salary). Join employees and departments.",
        "Find customers who have placed orders but whose total order amount doesn't match their total_spent field. Join customers and orders.",
        "Show each employee with their department name, manager name, and city. Self-join employees for manager lookup and join departments.",
        "Find products where the total ordered quantity exceeds current inventory. Join products, orders, and inventory.",
        "Show all sales with the employee name, product name, and sale amount, but only for sales in regions where the employee's department has an office. Join sales, employees, departments.",
        "Find users who have written reviews for products they never ordered. Join users, reviews, orders.",
        "For each customer, show their most expensive order, the product name, and the payment method used. Join customers, orders, products, payments.",
        "Find employees who manage at least 2 other employees. Self-join on manager_id.",
        "Show the average review rating for each product category. Join products and reviews, group by category.",
        "Find orders where the payment amount doesn't match the order total. Join orders and payments.",
        "Show each warehouse's total inventory value (quantity * product price). Join inventory and products.",
        "Find customers in the same city as at least one employee. Join customers and employees on city.",
        "For each product, show the number of unique customers who ordered it. Join products, orders, and customers.",
        "Find departments with no employees. Use a LEFT JOIN between departments and employees.",
        "Show the top-selling product in each sales region. Join sales and products, use GROUP BY with aggregation.",
        "Find employees whose salary is higher than the budget of any department divided by the number of employees in that department. Join employees and departments.",
        "Show orders where the customer's country is different from the product's sale region. Join orders, customers, and sales.",
        "Find products that are in inventory but have never been sold. Use LEFT JOIN between inventory/products and sales.",
        "For each employee, show the number of distinct products they've sold and the number of distinct regions. Join employees and sales.",
        "Find customers who have made payments with more than one payment method. Join customers, orders, and payments.",
        "Show each department's headcount, average salary, min salary, max salary, and salary range. Join employees with departments.",
        "Find the most reviewed product in each category. Join products and reviews, aggregate, and rank.",
        "Show transactions with the user's username and their activity status. Join transactions, and users on account_id = user id.",
        "Find employees hired in the same month as another employee in a different department.",
        "Show each customer's total order amount alongside the average for their country. Join customers and orders.",
        "Find products whose price is above the average price of their category. Use a subquery or join.",
        "Show all orders with the customer name, product name, order status, and payment status. Four-table join.",
    ]
    for p in join_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    # --- Category 4: Aggregation challenges (25 prompts) ---
    agg_prompts = [
        "Find departments where the average salary is more than 20% above the company average. Show department name and average salary.",
        "Show the number of orders per month and per status. Pivot-like query using CASE WHEN.",
        "Find the top 3 customers by total order amount. Show name, order count, and total amount.",
        "Calculate the standard deviation of salaries across all employees.",
        "Find products with an average review rating below 3.0 that have been ordered more than 5 times.",
        "Show the percentage of orders in each status category (shipped, delivered, pending, cancelled).",
        "Find the employee with the longest tenure (earliest hire_date) in each department.",
        "Calculate the total inventory value per warehouse and show warehouses above $10,000 total value.",
        "Find months where total sales exceeded the previous month by more than 50%.",
        "Show the distribution of review ratings (count per rating value 1-5) for each product.",
        "Find the city with the most employees and the city with the highest average salary. Are they the same?",
        "Calculate the Gini coefficient of employee salaries using SQL (show the formula steps in a CTE).",
        "Find customers who have spent more than 2 standard deviations above the mean total_spent.",
        "Show the ratio of Engineering salaries to Marketing salaries.",
        "Find the average order value by day of week (if extractable from order_date).",
        "Show departments ranked by total salary expenditure, with a running total column.",
        "Find products where the total reviews count is above average but the average rating is below average.",
        "Calculate the interquartile range (IQR) of product prices.",
        "Show each department's salary as a percentage of total company payroll.",
        "Find the most common order quantity and the product it's most associated with.",
        "Show a summary: total employees, total departments, avg salary, avg GPA of students, total orders.",
        "Find transactions where the amount is more than 3x the average transaction amount for that account.",
        "Show the growth rate of cumulative total_spent per customer signup month.",
        "Find the correlation between employee age and salary using SQL aggregation functions.",
        "Show products that have inventory in only one warehouse versus multiple warehouses.",
    ]
    for p in agg_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    # --- Category 5: CASE WHEN / conditional logic (20 prompts) ---
    case_prompts = [
        "Classify employees as 'Junior' (age < 28), 'Mid' (28-35), or 'Senior' (> 35). Show name, age, and classification.",
        "Create a salary band report: 'Low' (<70000), 'Medium' (70000-100000), 'High' (>100000). Count employees per band.",
        "Show each order with a 'size' label: 'Small' (total < 200), 'Medium' (200-500), 'Large' (> 500).",
        "Flag products as 'Low Stock' (stock < 30), 'Adequate' (30-100), or 'Overstocked' (> 100).",
        "For each customer, show 'Active' if they have orders in the last 6 months, 'Dormant' otherwise.",
        "Create a conditional join: show each employee's department budget only if their salary exceeds 80% of the department average.",
        "Show each review as 'Positive' (rating >= 4), 'Neutral' (3), or 'Negative' (< 3). Count per product.",
        "Calculate a bonus column: employees in Engineering get 15% bonus, Sales gets 10%, others get 5%. Show name, salary, bonus amount.",
        "Show each transaction with a flag indicating if it's above or below the account's average transaction amount.",
        "Classify customers by spending tier: 'Bronze' (<$1000), 'Silver' ($1000-$2000), 'Gold' ($2000-$3000), 'Platinum' (>$3000).",
        "Show each student with a letter grade based on GPA: A (>=3.7), B (>=3.0), C (>=2.0), D (>=1.0), F (<1.0).",
        "Create a report showing each department with columns for count of 'Junior', 'Mid', and 'Senior' employees.",
        "For each product, show the sentiment: 'Loved' if avg rating > 4, 'Mixed' if 2.5-4, 'Disliked' if < 2.5.",
        "Show orders with a priority flag: 'Rush' if quantity > 3 AND status = 'pending', 'Normal' otherwise.",
        "Calculate commission: 5% for orders under $200, 8% for $200-500, 12% for over $500. Show per order.",
        "Show each payment with a status label: 'On Time' if payment_date <= order_date + 30 days, 'Late' otherwise.",
        "Create a summary showing how many employees have 'above average', 'average' (within 10%), or 'below average' salary.",
        "Show products where the review sentiment (avg rating) contradicts sales volume (total quantity ordered).",
        "For each customer, show the percentage of their orders that are 'delivered' vs other statuses.",
        "Flag transactions as 'suspicious' if the amount is negative and > 2x the average debit for that account.",
    ]
    for p in case_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    # --- Category 6: Date/temporal queries (15 prompts) ---
    date_prompts = [
        "Find the average number of days between a customer's first and second order.",
        "Show monthly order counts for each quarter of the year.",
        "Find employees hired on a weekend (Saturday or Sunday), if any.",
        "Show the number of new customers per month (based on signup_date).",
        "Find the busiest day of the week for orders.",
        "Calculate each employee's tenure in years and months based on hire_date.",
        "Show orders placed within 7 days of a customer's signup_date.",
        "Find the month with the highest average order value.",
        "Show the time gap between consecutive transactions for each account.",
        "Find products that were reviewed within 7 days of being ordered.",
        "Show the year-over-year growth in total order amounts.",
        "Find customers who placed orders in at least 3 different months.",
        "Calculate the average review submission delay (days between order and review).",
        "Show the seasonal distribution of sales: Q1 (Jan-Mar), Q2 (Apr-Jun), Q3 (Jul-Sep), Q4 (Oct-Dec).",
        "Find employees whose hire anniversary falls in the current month.",
    ]
    for p in date_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    # --- Category 7: Advanced analytical (20 prompts) ---
    advanced_prompts = [
        "Write a query to detect potential duplicate customers (same first_name and city but different IDs).",
        "Find all employees who have no one reporting to them (leaf nodes in the org chart).",
        "Write a query that shows the Pareto principle: find the minimum number of customers that account for 80% of total order revenue.",
        "Find products where inventory is split across multiple warehouses — show the product, warehouses, and quantities.",
        "Write a query to detect anomalous transactions: amounts that are more than 2 standard deviations from the mean for that account.",
        "Find the most common 'path' through order statuses for each customer (e.g., pending -> shipped -> delivered).",
        "Write a query showing the customer lifetime value (CLV) for each customer: total_spent / months since signup.",
        "Find departments that are 'overstaffed' (headcount > budget/60000) or 'understaffed' (headcount < budget/120000).",
        "Write a pivot query showing each department as a column with total salary as the value.",
        "Find the optimal price point: the price range that generates the most total revenue (price * quantity ordered).",
        "Write a self-join to find all pairs of products that were ordered by the same customer.",
        "Find orders where the quantity * unit price doesn't match the order total (data inconsistency check).",
        "Write a query to compute the ABC classification of inventory: A (top 20% by value), B (next 30%), C (bottom 50%).",
        "Find the 'cross-sell' opportunities: products frequently ordered together by the same customer.",
        "Write a query showing the employee turnover rate by department (employees hired in a year / avg headcount).",
        "Find reviews that might be fake: same user_id reviewing the same product multiple times.",
        "Write a query to calculate the weighted average price of products, weighted by order quantity.",
        "Find the break-even point for each product: at what quantity do total sales exceed total inventory cost.",
        "Write a query showing the funnel: total customers -> customers with orders -> customers with delivered orders -> customers with reviews.",
        "Find the 'best value' products: highest average review rating per dollar of price.",
    ]
    for p in advanced_prompts:
        prompts.append({"prompt": p, "domain": "sql", "ground_truth": None})

    return prompts[:200]


# =============================================================================
# SOLIDITY PROMPTS — Contract specs requiring specific features
# =============================================================================

def generate_solidity_prompts() -> List[Dict]:
    """Generate 200 Solidity contract prompts requiring compilation."""
    prompts = []

    # --- Category 1: Token standards (30 prompts) ---
    token_prompts = [
        "Write an ERC20 token contract called 'MambaToken' with symbol 'MAMBA', 18 decimals, and an initial supply of 1,000,000 tokens minted to the deployer. Include transfer, approve, and transferFrom functions.",
        "Write a mintable ERC20 token where only the owner can mint new tokens. Include a mint function with an onlyOwner modifier.",
        "Write an ERC20 token with a burn function that allows any holder to burn their own tokens, reducing total supply.",
        "Write an ERC20 token with a maximum supply cap of 10,000,000 tokens. The mint function should revert if cap is exceeded.",
        "Write an ERC20 token with a 2% transfer tax. On each transfer, 2% is sent to a fee collector address and 98% to the recipient.",
        "Write an ERC20 token with pausable transfers. Only the owner can pause and unpause. Transfers revert when paused.",
        "Write an ERC20 token with a blacklist. The owner can add/remove addresses from the blacklist. Blacklisted addresses cannot send or receive tokens.",
        "Write an ERC20 token with snapshot functionality. The owner can take a snapshot, and balanceOfAt(address, snapshotId) returns the balance at that snapshot.",
        "Write a simple ERC721 NFT contract called 'MambaArt' with symbol 'MART'. Include mint, transferFrom, and ownerOf functions.",
        "Write an ERC721 contract with sequential token ID minting and a max supply of 1000.",
        "Write an ERC721 contract with metadata URI support (tokenURI function) that returns a base URI concatenated with the token ID.",
        "Write an ERC721 contract where only whitelisted addresses can mint during the first 24 hours after deployment.",
        "Write an ERC721 contract with royalties: include a royaltyInfo function per EIP-2981 that returns 5% royalty to the original creator.",
        "Write an ERC20 token with a vesting schedule. Tokens vest linearly over 12 months. Include claim() to withdraw vested tokens.",
        "Write an ERC20 token with delegation: holders can delegate voting power to another address. Include delegate() and getVotes() functions.",
        "Write a wrapped ETH (WETH) contract: deposit ETH to get tokens, withdraw tokens to get ETH back. 1:1 ratio.",
        "Write an ERC20 token with a transaction limit: no single transfer can exceed 1% of total supply.",
        "Write an ERC20 token with auto-liquidity: 1% of each transfer goes to a liquidity pool address.",
        "Write an ERC721A-style contract with batch minting: a function that mints N tokens in one transaction (gas optimized).",
        "Write an ERC1155 multi-token contract supporting both fungible (id=0, supply=1M) and non-fungible (id=1+, supply=1) tokens.",
        "Write an ERC20 token that implements permit() for gasless approvals (EIP-2612).",
        "Write a token contract that combines ERC20 and ERC721: fungible tokens and NFTs in the same contract.",
        "Write an ERC20 rebasing token: total supply increases by 1% daily, and all balances scale proportionally.",
        "Write an ERC20 token with a cooldown: each address can only transfer once every 60 seconds.",
        "Write an ERC721 contract with an auction-based minting: each mint costs the current price, which increases by 0.01 ETH per mint.",
        "Write a soulbound token (non-transferable ERC721). Override transfer functions to revert.",
        "Write an ERC20 token that tracks the top 10 holders in a sorted array, updated on each transfer.",
        "Write an ERC20 token with staking: holders can stake tokens and earn 10% APY, claimable at any time.",
        "Write an ERC20 token with a governance module: proposal creation, voting (1 token = 1 vote), and execution.",
        "Write an ERC20 token with multi-sig minting: minting requires approval from 2 out of 3 designated signers.",
    ]
    for p in token_prompts:
        prompts.append({"prompt": p, "domain": "solidity", "ground_truth": None})

    # --- Category 2: DeFi patterns (30 prompts) ---
    defi_prompts = [
        "Write an escrow contract: buyer deposits ETH, seller delivers (confirmed by buyer or arbiter), then funds release. Include dispute resolution with timeout.",
        "Write a simple DEX (decentralized exchange) pair contract with addLiquidity, removeLiquidity, and swap functions using the constant product formula (x*y=k).",
        "Write a staking pool contract where users stake an ERC20 token and earn rewards proportional to their share of the pool.",
        "Write a Dutch auction contract: starting price decreases over time. First bidder to accept the current price wins.",
        "Write a lending pool contract: users deposit ETH as collateral and can borrow up to 75% of their collateral value.",
        "Write a yield farming contract where users deposit LP tokens and earn a reward token at a fixed rate per block.",
        "Write a flash loan contract: users can borrow any amount with no collateral, but must repay within the same transaction. Charge a 0.1% fee.",
        "Write a price oracle contract that accepts price updates from whitelisted reporters and returns the median price.",
        "Write a vault contract that auto-compounds rewards: deposits earn yield, and the compound() function reinvests earnings.",
        "Write a bonding curve contract: token price increases linearly with supply. Buy and sell functions follow the curve.",
        "Write a liquidity mining contract with epoch-based rewards: each epoch (7 days) distributes a fixed reward pool proportionally.",
        "Write a multi-collateral vault: users can deposit ETH or any approved ERC20 as collateral, with different collateral ratios.",
        "Write a governance timelock contract: approved proposals can only execute after a 48-hour delay.",
        "Write an options contract: users can buy call options on ETH at a strike price, expiring at a specific block.",
        "Write a perpetual swap contract with funding rates: longs pay shorts (or vice versa) based on the mark-index price spread.",
        "Write a token bridge contract: lock tokens on this chain and emit an event for the relayer to mint on the destination chain.",
        "Write a fee distributor contract: collects protocol fees and distributes them proportionally to stakers.",
        "Write a liquidation bot contract: anyone can liquidate undercollateralized positions and earn a 5% liquidation bonus.",
        "Write a limit order contract: users set price thresholds, and keepers execute orders when conditions are met.",
        "Write an insurance pool contract: users pay premiums, and verified claims get paid from the pool. Surplus earns yield.",
        "Write a rebasing vault: deposit tokens, the vault earns yield, and share price increases. No new shares minted.",
        "Write a token vesting contract with cliff: no tokens vest for 6 months, then linear vesting over 18 months.",
        "Write a DAO treasury contract: proposals require 10% quorum and 60% approval to pass. Execution is timelocked.",
        "Write a batch auction contract: users submit sealed bids during a commit phase, reveal during reveal phase, clearing price determined.",
        "Write a gas-efficient airdrop contract that distributes tokens to multiple addresses in a single transaction using a Merkle proof.",
        "Write a recurring payment contract: payer authorizes, and payee can claim a fixed amount every 30 days.",
        "Write a strategy vault pattern: the vault holds assets and delegates investment to a replaceable strategy contract.",
        "Write a multi-sig wallet requiring 2-of-3 signatures to execute any transaction. Include propose, approve, and execute functions.",
        "Write a token locker contract: users lock tokens for a specified duration. Early withdrawal incurs a 10% penalty.",
        "Write a lottery contract: users buy tickets with ETH, a random winner is selected using blockhash, and they receive the pot minus a 5% fee.",
    ]
    for p in defi_prompts:
        prompts.append({"prompt": p, "domain": "solidity", "ground_truth": None})

    # --- Category 3: Access control & security (25 prompts) ---
    access_prompts = [
        "Write a contract with role-based access control: ADMIN, MINTER, and PAUSER roles. Include grantRole, revokeRole, and modifier onlyRole.",
        "Write a contract implementing the checks-effects-interactions pattern for a withdrawal function that prevents reentrancy.",
        "Write an Ownable contract with two-step ownership transfer: the new owner must accept the transfer.",
        "Write a contract with a time-locked admin function: the admin can only call sensitive functions after a 24-hour cooldown.",
        "Write a proxy contract implementing the minimal proxy pattern (EIP-1167) for cheap cloning.",
        "Write a contract with rate limiting: each address can call a function at most 5 times per hour.",
        "Write a contract with emergency stop (circuit breaker): the guardian can freeze all operations.",
        "Write a contract with a whitelist that uses a Merkle tree for gas-efficient verification.",
        "Write an upgradeable contract using the transparent proxy pattern: proxy delegates to implementation, admin can upgrade.",
        "Write a contract with reentrancy guard using a mutex lock. The modifier should prevent nested calls.",
        "Write a contract that implements pull payments: instead of pushing ETH to recipients, it lets them withdraw.",
        "Write a contract with role hierarchy: SUPER_ADMIN can manage ADMIN, ADMIN can manage OPERATOR.",
        "Write a contract with a dead man's switch: if the owner doesn't check in within 90 days, a backup address gains control.",
        "Write a contract with spending limits: each authorized spender has a daily limit that resets at midnight UTC.",
        "Write a contract implementing commit-reveal for fair random selection: users commit a hash, then reveal their value.",
        "Write a factory contract that deploys new instances of a child contract with CREATE2 for deterministic addresses.",
        "Write a contract with access control based on token holding: only addresses holding >= 100 tokens can call privileged functions.",
        "Write a contract with a multi-phase launch: Phase 1 (whitelist only), Phase 2 (public), Phase 3 (closed). Only owner can advance phases.",
        "Write a contract with delegatecall safety: a library contract that can be safely called via delegatecall without storage collisions.",
        "Write a contract implementing the withdrawal pattern with a mapping of pending withdrawals and a separate withdraw function.",
        "Write a contract with a cooldown period between sensitive operations: owner must wait 3 days between parameter changes.",
        "Write a contract where admin actions require confirmation from a separate guardian address within 24 hours.",
        "Write a contract that self-destructs after a specified timestamp, sending remaining ETH to the owner.",
        "Write a contract with permit-based access: off-chain signatures authorize on-chain actions (EIP-712 typed data).",
        "Write a contract that limits gas consumption per call to prevent griefing attacks on unbounded loops.",
    ]
    for p in access_prompts:
        prompts.append({"prompt": p, "domain": "solidity", "ground_truth": None})

    # --- Category 4: Data structures & patterns (25 prompts) ---
    pattern_prompts = [
        "Write a contract implementing an on-chain linked list with insert, remove, and iterate functions.",
        "Write a contract implementing a priority queue using a binary heap stored in an array.",
        "Write a contract implementing a simple key-value store with get, set, and delete operations.",
        "Write a contract implementing an iterable mapping: a mapping that can be enumerated.",
        "Write a contract implementing a circular buffer for storing the last 100 events.",
        "Write a contract implementing a simple state machine with states: Created, Active, Paused, Terminated.",
        "Write a contract implementing an order book: users place buy/sell orders, matching engine fills them.",
        "Write a contract implementing a simple voting system: create proposals, vote with token weight, tally results.",
        "Write a contract implementing a subscription model: users pay monthly, access expires if not renewed.",
        "Write a contract that stores a Merkle tree root and verifies inclusion proofs.",
        "Write a contract implementing a registry pattern: register name->address mappings with expiration.",
        "Write a contract implementing a simple AMM with concentrated liquidity in a single price range.",
        "Write a contract implementing a queue (FIFO) for processing requests in order.",
        "Write a contract implementing a reputation system: users earn/lose points, with a decay mechanism.",
        "Write a contract implementing a simple NFT marketplace: list, buy, cancel listing, with royalties.",
        "Write a contract implementing a payment splitter that divides incoming ETH among multiple payees by shares.",
        "Write a contract implementing a coupon/voucher system: admin creates vouchers with codes, users redeem them.",
        "Write a contract implementing a simple prediction market: binary outcomes, users bet, oracle resolves.",
        "Write a contract implementing a tip jar: anyone can send tips, the owner can withdraw, tips are tracked per sender.",
        "Write a contract implementing gas-efficient batch transfers of ETH to multiple addresses.",
        "Write a contract implementing a simple token swap: users deposit token A, receive token B at a fixed rate.",
        "Write a contract implementing an allowance system: parent grants child a daily spending allowance in ETH.",
        "Write a contract implementing a simple DAO: propose, vote, execute, with minimum quorum and vote threshold.",
        "Write a contract implementing a time-weighted average price (TWAP) oracle that accumulates price * time.",
        "Write a contract implementing a verifiable random function (VRF) consumer that requests and receives randomness.",
    ]
    for p in pattern_prompts:
        prompts.append({"prompt": p, "domain": "solidity", "ground_truth": None})

    # --- Category 5: Gas optimization & events (20 prompts) ---
    gas_prompts = [
        "Write a gas-optimized ERC20 token using packed storage: pack balance and allowance into fewer slots.",
        "Write a contract that uses events extensively: emit Transfer, Approval, Mint, Burn events with indexed parameters.",
        "Write a contract using assembly (inline Yul) for an efficient memory copy operation.",
        "Write a contract that uses custom errors (error InsufficientBalance(uint256 requested, uint256 available)) instead of require strings for gas savings.",
        "Write a contract that uses immutable variables for deploy-time constants.",
        "Write a contract using unchecked arithmetic where overflow is impossible (e.g., loop counter).",
        "Write a contract that uses calldata instead of memory for function parameters where the data isn't modified.",
        "Write a contract that packs multiple boolean flags into a single uint256 using bitwise operations.",
        "Write a contract that minimizes SSTORE operations by batching state changes.",
        "Write a contract with efficient array operations: avoid copying, use in-place swaps for deletion.",
        "Write a contract that uses mapping(address => uint256) instead of arrays for O(1) lookups.",
        "Write a contract with a tight loop that processes up to 100 elements without exceeding the block gas limit.",
        "Write a contract that uses bytes32 instead of string for short identifiers (< 32 bytes).",
        "Write a contract with fallback and receive functions that efficiently handle direct ETH transfers.",
        "Write a contract using structs with packed storage layout to minimize slot usage.",
        "Write a contract that uses create2 to predict deployment addresses before deploying.",
        "Write a contract that emits structured event logs for off-chain indexing: an event for every state change with all relevant data.",
        "Write a contract with a gas-efficient batch approve function that sets allowances for multiple spenders.",
        "Write a contract using the diamond pattern (EIP-2535) with multiple facets sharing storage.",
        "Write a contract that implements lazy evaluation: expensive computations are deferred until their results are needed.",
    ]
    for p in gas_prompts:
        prompts.append({"prompt": p, "domain": "solidity", "ground_truth": None})

    # --- Category 6: ZK-relevant contracts (20 prompts) ---
    zk_prompts = [
        "Write a contract that verifies a simple zk-SNARK proof. Include a verifyProof function that checks pairing equations (mock the pairing with simple math).",
        "Write a contract implementing a commitment scheme: users submit hash(secret || value), then reveal the secret and value.",
        "Write a contract that verifies Merkle inclusion proofs for a state tree with bytes32 leaves.",
        "Write a contract implementing a simple mixer: users deposit 1 ETH with a commitment, withdraw with a nullifier (preventing double-spend).",
        "Write a contract that stores computation results with their proof hashes, allowing verification without re-execution.",
        "Write a contract implementing a verifiable computation marketplace: post computation tasks, submit results with proofs.",
        "Write a contract that implements a simple hash chain for sequential proof verification.",
        "Write a contract for a private voting system using commitments: vote is hidden until reveal phase.",
        "Write a contract implementing accumulator-based set membership proof verification.",
        "Write a contract that batches multiple state transitions into a single proof verification (rollup-style).",
        "Write a contract implementing a simple state channel: open, update (with signatures), close, dispute.",
        "Write a contract that verifies polynomial commitments (KZG-style) using simple modular arithmetic.",
        "Write a contract implementing a nullifier registry to prevent double-spending of commitments.",
        "Write a contract for a sealed-bid auction using hash commitments for bid privacy.",
        "Write a contract implementing a simple plasma chain: deposit, submit blocks (as Merkle roots), exit with proof.",
        "Write a contract that verifies BLS signatures (simplified) for aggregate signature verification.",
        "Write a contract implementing a verifiable delay function (VDF) result verification.",
        "Write a contract for confidential token transfers: amounts hidden via Pedersen commitments, range proofs verified.",
        "Write a contract implementing a simple optimistic rollup: submit batches, fraud proof period, finalization.",
        "Write a contract that stores IPFS content hashes (CIDs) with associated metadata and access control.",
    ]
    for p in zk_prompts:
        prompts.append({"prompt": p, "domain": "solidity", "ground_truth": None})

    return prompts[:200]


# =============================================================================
# SEC/FINANCE PROMPTS — Structured analysis requiring specific metrics
# =============================================================================

def generate_sec_finance_prompts() -> List[Dict]:
    """Generate 200 SEC/Finance analysis prompts."""
    prompts = []

    companies = [
        ("Apple Inc.", "AAPL", "technology"),
        ("Microsoft Corporation", "MSFT", "technology"),
        ("Amazon.com Inc.", "AMZN", "e-commerce/cloud"),
        ("Alphabet Inc.", "GOOGL", "technology/advertising"),
        ("Tesla Inc.", "TSLA", "automotive/energy"),
        ("JPMorgan Chase & Co.", "JPM", "banking"),
        ("Johnson & Johnson", "JNJ", "healthcare"),
        ("Walmart Inc.", "WMT", "retail"),
        ("ExxonMobil Corporation", "XOM", "energy"),
        ("Pfizer Inc.", "PFE", "pharmaceuticals"),
        ("NVIDIA Corporation", "NVDA", "semiconductors"),
        ("Meta Platforms Inc.", "META", "social media/technology"),
        ("Berkshire Hathaway Inc.", "BRK.B", "conglomerate"),
        ("Visa Inc.", "V", "financial services"),
        ("Procter & Gamble Co.", "PG", "consumer goods"),
        ("UnitedHealth Group Inc.", "UNH", "healthcare/insurance"),
        ("The Coca-Cola Company", "KO", "beverages"),
        ("Intel Corporation", "INTC", "semiconductors"),
        ("Disney (The Walt Disney Company)", "DIS", "entertainment"),
        ("Netflix Inc.", "NFLX", "streaming/entertainment"),
    ]

    years = ["2021", "2022", "2023"]

    # --- Category 1: Income statement analysis (40 prompts) ---
    for i in range(40):
        co = companies[i % len(companies)]
        yr = years[i % len(years)]
        templates = [
            f"Analyze {co[0]} ({co[1]})'s income statement for fiscal year {yr}. Calculate and discuss: (1) gross margin, (2) operating margin, (3) net profit margin. Compare to industry averages for the {co[2]} sector.",
            f"For {co[0]}'s {yr} annual report, compute the year-over-year revenue growth rate and explain the key drivers. Include specific dollar amounts and percentages.",
            f"Analyze {co[0]}'s cost structure for {yr}: break down COGS, SG&A, R&D as percentages of revenue. Identify the largest cost component and discuss trends.",
            f"Calculate {co[0]}'s EBITDA and EBITDA margin for {yr}. Discuss how depreciation and amortization impact the difference between operating income and EBITDA.",
            f"Analyze {co[0]}'s earnings per share (EPS) for {yr}. Discuss diluted vs basic EPS, the impact of share buybacks, and compare to analyst consensus estimates.",
        ]
        prompts.append({
            "prompt": templates[i % len(templates)],
            "domain": "sec_finance",
            "ground_truth": None
        })

    # --- Category 2: Balance sheet analysis (35 prompts) ---
    for i in range(35):
        co = companies[i % len(companies)]
        yr = years[i % len(years)]
        templates = [
            f"Analyze {co[0]}'s balance sheet as of {yr} year-end. Calculate: (1) current ratio, (2) quick ratio, (3) debt-to-equity ratio. Assess the company's liquidity and solvency position.",
            f"Evaluate {co[0]}'s working capital management for {yr}. Calculate days sales outstanding (DSO), days inventory outstanding (DIO), and days payable outstanding (DPO). Compute the cash conversion cycle.",
            f"Analyze {co[0]}'s capital structure at {yr} year-end. Break down total debt (short-term vs long-term), equity components, and calculate the weighted average cost of capital (WACC) assuming a market risk premium of 6%.",
            f"Assess {co[0]}'s asset quality for {yr}: compute return on assets (ROA), asset turnover ratio, and discuss goodwill as a percentage of total assets. Flag any impairment risks.",
            f"Evaluate {co[0]}'s shareholder equity components for {yr}: retained earnings trend, treasury stock, accumulated other comprehensive income. Calculate book value per share.",
        ]
        prompts.append({
            "prompt": templates[i % len(templates)],
            "domain": "sec_finance",
            "ground_truth": None
        })

    # --- Category 3: Cash flow analysis (30 prompts) ---
    for i in range(30):
        co = companies[i % len(companies)]
        yr = years[i % len(years)]
        templates = [
            f"Analyze {co[0]}'s cash flow statement for {yr}. Compute free cash flow (FCF), FCF yield, and FCF per share. Discuss the quality of earnings by comparing net income to operating cash flow.",
            f"For {co[0]} in {yr}, calculate the cash flow from operations to net income ratio. Discuss any significant non-cash adjustments and changes in working capital.",
            f"Evaluate {co[0]}'s capital allocation for {yr}: break down capital expenditures, acquisitions, share buybacks, and dividends as a percentage of operating cash flow.",
            f"Assess {co[0]}'s cash position for {yr}: compute the cash burn/generation rate, runway (months of cash remaining at current burn rate), and discuss cash management strategy.",
            f"Analyze {co[0]}'s investing activities for {yr}: capital expenditures vs depreciation (maintenance vs growth capex), acquisition spending, and returns on invested capital (ROIC).",
        ]
        prompts.append({
            "prompt": templates[i % len(templates)],
            "domain": "sec_finance",
            "ground_truth": None
        })

    # --- Category 4: Ratio analysis & valuation (30 prompts) ---
    for i in range(30):
        co = companies[i % len(companies)]
        yr = years[i % len(years)]
        templates = [
            f"Perform a comprehensive ratio analysis of {co[0]} for {yr}: profitability (ROE, ROA, margins), liquidity (current, quick), leverage (D/E, interest coverage), efficiency (asset turnover, inventory turnover).",
            f"Estimate {co[0]}'s intrinsic value using a discounted cash flow (DCF) model with {yr} as the base year. Assume 8% revenue growth declining to 3% terminal growth, 15% discount rate. Show all steps.",
            f"Calculate {co[0]}'s DuPont decomposition for {yr}: break ROE into profit margin × asset turnover × equity multiplier. Identify the primary driver of ROE.",
            f"Compare {co[0]}'s valuation multiples (P/E, P/S, EV/EBITDA, P/B) for {yr} against the {co[2]} sector average. Is the stock overvalued or undervalued?",
            f"Calculate {co[0]}'s economic value added (EVA) for {yr}: NOPAT minus (invested capital × WACC). Discuss whether the company creates or destroys shareholder value.",
            f"Analyze {co[0]}'s dividend policy for {yr}: dividend yield, payout ratio, dividend coverage ratio, dividend growth rate over the past 3 years.",
        ]
        prompts.append({
            "prompt": templates[i % len(templates)],
            "domain": "sec_finance",
            "ground_truth": None
        })

    # --- Category 5: Risk analysis & SEC filings (30 prompts) ---
    for i in range(30):
        co = companies[i % len(companies)]
        yr = years[i % len(years)]
        templates = [
            f"Analyze the risk factors section of {co[0]}'s {yr} 10-K filing. Identify the top 5 material risks, categorize them (operational, financial, regulatory, market, strategic), and assess their potential financial impact.",
            f"Evaluate {co[0]}'s regulatory risk exposure for {yr}: identify key regulatory requirements, recent enforcement actions, and estimated compliance costs. Discuss antitrust, data privacy, and environmental regulations.",
            f"Assess {co[0]}'s foreign exchange risk for {yr}: estimate revenue exposure by currency, discuss hedging strategies, and calculate the potential impact of a 10% USD strengthening on revenue.",
            f"Analyze {co[0]}'s credit risk for {yr}: evaluate debt ratings, interest coverage ratio, debt maturity schedule, and refinancing risk. Discuss the impact of rising interest rates.",
            f"Review {co[0]}'s {yr} 10-K for off-balance sheet obligations: operating leases (pre/post ASC 842), purchase commitments, guarantees. Calculate the adjusted leverage ratio including these items.",
            f"Evaluate {co[0]}'s concentration risk for {yr}: revenue concentration by customer, geography, and product/service. Assess supply chain dependencies and single points of failure.",
        ]
        prompts.append({
            "prompt": templates[i % len(templates)],
            "domain": "sec_finance",
            "ground_truth": None
        })

    # --- Category 6: Comparative & sector analysis (35 prompts) ---
    sector_pairs = [
        (("Apple Inc.", "AAPL"), ("Microsoft Corporation", "MSFT"), "technology"),
        (("Amazon.com Inc.", "AMZN"), ("Walmart Inc.", "WMT"), "retail"),
        (("JPMorgan Chase & Co.", "JPM"), ("Visa Inc.", "V"), "financial services"),
        (("Tesla Inc.", "TSLA"), ("ExxonMobil Corporation", "XOM"), "energy transition"),
        (("NVIDIA Corporation", "NVDA"), ("Intel Corporation", "INTC"), "semiconductors"),
        (("Johnson & Johnson", "JNJ"), ("Pfizer Inc.", "PFE"), "healthcare"),
        (("Netflix Inc.", "NFLX"), ("Disney", "DIS"), "entertainment"),
    ]
    for i in range(35):
        pair = sector_pairs[i % len(sector_pairs)]
        yr = years[i % len(years)]
        templates = [
            f"Compare {pair[0][0]} ({pair[0][1]}) and {pair[1][0]} ({pair[1][1]}) for {yr}: revenue, margins, growth rate, and valuation multiples. Which company offers better value?",
            f"Perform a SWOT analysis comparing {pair[0][0]} and {pair[1][0]} in the {pair[2]} sector for {yr}. Include specific financial metrics to support each point.",
            f"Compare the capital allocation strategies of {pair[0][0]} and {pair[1][0]} for {yr}: R&D spending, capex, acquisitions, buybacks, and dividends as % of revenue.",
            f"Analyze the competitive dynamics between {pair[0][0]} and {pair[1][0]} in {yr}: market share, revenue growth, innovation spending, and moat characteristics.",
            f"Compare the financial health of {pair[0][0]} and {pair[1][0]} for {yr} using Altman Z-Score and Piotroski F-Score. Identify the financially stronger company.",
        ]
        prompts.append({
            "prompt": templates[i % len(templates)],
            "domain": "sec_finance",
            "ground_truth": None
        })

    return prompts[:200]


# =============================================================================
# ENGLISH PROMPTS — Technical writing on ZK/ML/blockchain topics
# =============================================================================

def generate_english_prompts() -> List[Dict]:
    """Generate 150 English writing prompts requiring structured, coherent prose."""
    prompts = []

    # --- Category 1: ZK/cryptography explanations (30 prompts) ---
    zk_topics = [
        "Explain how zero-knowledge proofs work to a software engineer who has no cryptography background. Cover: what they prove, why they're useful, and give a concrete example (like proving you know a password without revealing it). Structure your response with an introduction, mechanism, example, and conclusion.",
        "Describe the difference between zk-SNARKs and zk-STARKs. Compare their trust assumptions, proof sizes, verification times, and use cases. Use specific technical details but explain jargon.",
        "Explain the concept of a trusted setup in zk-SNARKs. Why is it necessary? What are the risks? How do ceremonies like Powers of Tau mitigate them? Write for a technically literate audience.",
        "Describe how recursive proof composition works in zero-knowledge systems. Explain why it matters for scalability, and give an example of how a blockchain rollup uses it.",
        "Explain the relationship between polynomial commitments and zero-knowledge proofs. Cover KZG commitments, their role in proof systems, and why polynomial representation matters.",
        "Describe how zero-knowledge proofs enable private transactions on public blockchains. Compare the approaches of Zcash (Sapling), Tornado Cash, and Aztec Protocol.",
        "Explain what a zkEVM is and why building one is technically challenging. Discuss the tradeoffs between different approaches (type 1, 2, 3, 4 zkEVMs).",
        "Describe the concept of verifiable computation: how can a weak device verify that a powerful server performed a computation correctly? Explain the role of interactive and non-interactive proofs.",
        "Explain homomorphic encryption and its relationship to zero-knowledge proofs. Compare partially homomorphic, somewhat homomorphic, and fully homomorphic encryption.",
        "Describe how Merkle trees enable efficient data verification. Explain their structure, proof generation, proof verification, and applications in blockchains and certificate transparency.",
        "Explain the Fiat-Shamir heuristic: how it converts interactive proofs to non-interactive ones. Discuss security assumptions and potential pitfalls.",
        "Describe the role of elliptic curve cryptography in zero-knowledge proof systems. Explain pairings, the discrete log problem, and why specific curves are chosen.",
        "Explain what a commitment scheme is and give three examples: hash commitments, Pedersen commitments, and polynomial commitments. Compare their properties.",
        "Describe the concept of proof aggregation: how multiple proofs can be combined into one. Explain why this matters for blockchain scalability.",
        "Explain the arithmetization process: how arbitrary computations are converted into algebraic constraints that ZK proof systems can work with.",
        "Describe how zero-knowledge proofs can be used for identity verification: proving you're over 18 without revealing your birthday, or proving citizenship without revealing your passport.",
        "Explain the security properties of zero-knowledge proofs: completeness, soundness, and zero-knowledge. Give intuitive explanations of each.",
        "Describe the PLONK proof system: its universal trusted setup, custom gates, and why it became popular. Compare to Groth16.",
        "Explain how ZK rollups achieve both scalability and security. Describe the proof generation, verification, and data availability aspects.",
        "Describe the concept of a verifiable random function (VRF) and its applications in blockchain consensus, randomness beacons, and privacy.",
        "Explain what witness generation means in ZK proof systems. Describe the computation trace, constraint satisfaction, and why witness size matters.",
        "Describe the state of ZK hardware acceleration: FPGAs, ASICs, and GPUs for proof generation. Discuss the computational bottlenecks.",
        "Explain the concept of a ZK coprocessor: how it extends blockchain capabilities by proving off-chain computations.",
        "Describe how ZK proofs enable cross-chain bridges with strong security guarantees. Compare to multisig and optimistic approaches.",
        "Explain the Sumcheck protocol and its role as a building block in proof systems like Lasso and Jolt.",
        "Describe the Nova folding scheme for incrementally verifiable computation. Explain how it differs from recursive SNARKs.",
        "Explain what a lookup argument is in ZK proof systems and why it's important for efficient arithmetization.",
        "Describe the concept of composable privacy: how ZK proofs enable private state transitions in smart contracts.",
        "Explain the tradeoffs in proof system design: prover time, verifier time, proof size, trust assumptions, and universality.",
        "Describe how formal verification of ZK circuits prevents soundness bugs. Discuss recent vulnerabilities and lessons learned.",
    ]
    for p in zk_topics:
        prompts.append({"prompt": p, "domain": "english", "ground_truth": None})

    # --- Category 2: ML/AI explanations (30 prompts) ---
    ml_topics = [
        "Explain the transformer architecture to someone who understands basic neural networks but hasn't studied attention mechanisms. Cover self-attention, multi-head attention, positional encoding, and why transformers replaced RNNs.",
        "Describe the training process of large language models: pretraining on web text, supervised fine-tuning, RLHF, and constitutional AI. Explain what each stage teaches the model.",
        "Explain the concept of quantization in neural networks: why it's done, the tradeoffs, and the difference between post-training quantization and quantization-aware training. Include specific bit-widths (INT8, INT4, binary).",
        "Describe the Mamba architecture (selective state space model): how it differs from transformers, its linear-time complexity, and when it's advantageous. Compare to attention-based models.",
        "Explain reinforcement learning from human feedback (RLHF): the reward model, PPO optimization, and why it produces more aligned outputs than supervised fine-tuning alone.",
        "Describe the concept of model distillation: how a large 'teacher' model trains a smaller 'student' model. Discuss logit matching, feature matching, and when distillation works well.",
        "Explain the scaling laws of neural networks: how performance improves with model size, data size, and compute. Discuss the Chinchilla optimal ratio and its implications.",
        "Describe the difference between encoder, decoder, and encoder-decoder transformer architectures. Give examples of each (BERT, GPT, T5) and their ideal use cases.",
        "Explain what LoRA (Low-Rank Adaptation) is and why it's important for fine-tuning large models. Describe the mathematical formulation and practical benefits.",
        "Describe the concept of mixture of experts (MoE): how routing works, why it enables larger models at lower compute cost, and the challenges (load balancing, routing collapse).",
        "Explain the Group Relative Policy Optimization (GRPO) algorithm: how it differs from PPO, why it removes the need for a critic model, and its advantages for mathematical reasoning.",
        "Describe how tokenization works in language models: BPE, WordPiece, and SentencePiece. Explain why tokenization affects model performance on arithmetic and code.",
        "Explain the concept of emergent abilities in large language models: what they are, whether they're real or a measurement artifact, and implications for AI safety.",
        "Describe the attention mechanism mathematically: Q, K, V matrices, scaled dot-product, softmax, and why the scaling factor 1/sqrt(d_k) is needed.",
        "Explain how diffusion models generate images: the forward noising process, reverse denoising, and the role of the neural network. Compare to GANs and VAEs.",
        "Describe the concept of integer-only inference in neural networks: replacing all floating-point operations with integer arithmetic. Explain why this matters for hardware efficiency and ZK compatibility.",
        "Explain curriculum learning: training a model on progressively harder examples. Describe the theoretical motivation, implementation strategies, and empirical results.",
        "Describe the BitNet architecture: ternary weights {-1, 0, 1}, how training works with straight-through estimators, and the implications for inference efficiency.",
        "Explain flash attention: how it reduces memory from O(N^2) to O(N) by tiling the computation. Describe the IO-awareness aspect and hardware considerations.",
        "Describe the concept of constitutional AI: how models can be trained with a set of principles instead of human feedback for each example.",
        "Explain how state space models (SSMs) process sequences: the continuous-time formulation, discretization, and the parallel scan algorithm for efficient training.",
        "Describe the challenges of training on synthetic data: distribution collapse, mode collapse, model autophagy disorder, and strategies to mitigate them.",
        "Explain the concept of in-context learning: how language models can perform new tasks from examples in the prompt without gradient updates. Discuss theories for why it works.",
        "Describe activation quantization vs weight quantization: different strategies, accuracy tradeoffs, and hardware implications. Discuss mixed-precision approaches.",
        "Explain the concept of data contamination in language model benchmarks: how training data can leak into evaluation sets and strategies to detect/prevent it.",
        "Describe the concept of model merging: combining weights from multiple fine-tuned models. Cover TIES, DARE, and task arithmetic approaches.",
        "Explain how chain-of-thought prompting improves reasoning: the mechanism, when it helps, when it doesn't, and the connection to scratchpad training.",
        "Describe the concept of speculative decoding: how a small draft model accelerates inference from a large model. Explain the acceptance criterion.",
        "Explain the difference between supervised fine-tuning (SFT) and pretraining in terms of data requirements, learning objectives, and model behavior changes.",
        "Describe the RLVR (Reinforcement Learning with Verifiable Rewards) paradigm: using programmatic verifiers instead of human feedback for domains like math, code, and SQL.",
    ]
    for p in ml_topics:
        prompts.append({"prompt": p, "domain": "english", "ground_truth": None})

    # --- Category 3: Blockchain/Web3 explanations (25 prompts) ---
    blockchain_topics = [
        "Explain the Ethereum Virtual Machine (EVM): how it executes smart contracts, the concept of gas, and why it uses a stack-based architecture.",
        "Describe the transition from Proof of Work to Proof of Stake in Ethereum (The Merge). Compare energy consumption, security model, and validator economics.",
        "Explain how decentralized exchanges (DEXs) work: the automated market maker model, liquidity pools, impermanent loss, and the constant product formula.",
        "Describe the concept of MEV (Maximal Extractable Value): what it is, how searchers extract it, and its impact on users. Discuss Flashbots and PBS as mitigations.",
        "Explain Layer 2 scaling solutions: rollups (optimistic and ZK), state channels, and plasma. Compare their security assumptions and performance tradeoffs.",
        "Describe how decentralized identity (DID) systems work: self-sovereign identity, verifiable credentials, and the role of blockchain as a trust anchor.",
        "Explain the concept of account abstraction (ERC-4337): how it improves UX, enables smart contract wallets, and the role of bundlers and paymasters.",
        "Describe the oracle problem in blockchain: how smart contracts access off-chain data. Compare Chainlink, UMA, and API3 approaches.",
        "Explain the concept of restaking (EigenLayer): how it extends Ethereum's security to other protocols. Discuss risks and potential for cascading failures.",
        "Describe the concept of data availability in blockchain: why it matters for rollups, how DAS (Data Availability Sampling) works, and the role of Danksharding.",
        "Explain how NFTs work technically: the ERC-721 standard, metadata storage (on-chain vs IPFS vs Arweave), and why token ownership is separate from content ownership.",
        "Describe the concept of composability in DeFi: how protocols build on each other, the benefits of open standards, and the risks of dependency chains.",
        "Explain the CAP theorem as it applies to blockchain: consistency, availability, partition tolerance, and why different chains make different tradeoffs.",
        "Describe the concept of tokenomics: supply mechanics (inflation, deflation, rebasing), utility, governance, and how token design affects protocol sustainability.",
        "Explain how cross-chain bridges work: lock-and-mint, burn-and-mint, and atomic swaps. Discuss the security risks demonstrated by bridge hacks.",
        "Describe the concept of programmable money: how smart contracts enable conditional transfers, streaming payments, and automated financial agreements.",
        "Explain how decentralized governance works: token voting, quadratic voting, conviction voting, and the challenges of plutocracy and voter apathy.",
        "Describe the concept of modular blockchains: separating execution, settlement, data availability, and consensus into specialized layers.",
        "Explain the concept of intent-based transactions: users specify desired outcomes, solvers find optimal execution paths. Discuss UniswapX and other implementations.",
        "Describe the economics of Proof of Stake: staking yields, slashing conditions, validator selection, and the relationship between stake concentration and decentralization.",
        "Explain how blockchain-based supply chain tracking works: provenance, authenticity verification, and the challenges of the 'oracle problem' at physical-digital boundaries.",
        "Describe the concept of sovereign rollups: rollups that settle on a data availability layer instead of a smart contract chain. Compare to smart contract rollups.",
        "Explain the concept of a decentralized autonomous organization (DAO): governance, treasury management, and the legal/regulatory landscape.",
        "Describe the concept of liquid staking: staking derivatives (stETH, rETH), their role in DeFi composability, and the risks of depeg scenarios.",
        "Explain how blockchain timestamping works and its applications in intellectual property, scientific publishing, and audit trails.",
    ]
    for p in blockchain_topics:
        prompts.append({"prompt": p, "domain": "english", "ground_truth": None})

    # --- Category 4: Comparative essays (20 prompts) ---
    comparison_topics = [
        "Compare centralized and decentralized exchanges: security models, performance, user experience, regulatory compliance, and custody tradeoffs. Write a balanced analysis.",
        "Compare Proof of Work and Proof of Stake consensus mechanisms: energy efficiency, security guarantees, decentralization, and economic incentives.",
        "Compare the RISC-V and ARM instruction set architectures for ML inference: performance, power efficiency, extensibility, and ecosystem maturity.",
        "Compare transformer and state-space model architectures for sequence modeling: training efficiency, inference cost, long-context performance, and hardware utilization.",
        "Compare supervised fine-tuning and reinforcement learning for teaching language models to follow instructions: data requirements, output quality, and alignment properties.",
        "Compare optimistic and zero-knowledge rollups: finality time, cost, trust assumptions, EVM compatibility, and developer experience.",
        "Compare public and private blockchains for enterprise use: security, privacy, performance, governance, and regulatory compliance.",
        "Compare FPGA and ASIC accelerators for ZK proof generation: development cost, flexibility, performance, and time-to-market.",
        "Compare model distillation and quantization as approaches to making neural networks more efficient: accuracy-speed tradeoff, implementation complexity, and hardware requirements.",
        "Compare federated learning and differential privacy as approaches to privacy-preserving machine learning: threat models, performance impact, and practical deployments.",
        "Compare the Solidity and Rust programming languages for smart contract development: safety guarantees, developer experience, tooling, and ecosystem.",
        "Compare full fine-tuning, LoRA, and prompt tuning for adapting language models: parameter efficiency, task performance, and computational cost.",
        "Compare Ethereum, Solana, and Cosmos as platforms for decentralized applications: consensus, throughput, finality, developer tools, and ecosystem.",
        "Compare batch normalization and layer normalization in neural networks: when each is preferred, their effect on training dynamics, and behavior at inference time.",
        "Compare software-based and hardware-based approaches to neural network acceleration: GPUs, TPUs, neuromorphic chips, and photonic computing.",
        "Compare the retrieval-augmented generation (RAG) and fine-tuning approaches for domain-specific language models: cost, freshness, accuracy, and maintenance.",
        "Compare the security models of custodial and non-custodial cryptocurrency wallets: threat vectors, recovery options, and user experience tradeoffs.",
        "Compare Adam, SGD with momentum, and LAMB optimizers for training neural networks: convergence speed, generalization, and hyperparameter sensitivity.",
        "Compare the approaches of OpenAI, Anthropic, and DeepMind to AI safety: technical methods, organizational structure, and published research.",
        "Compare lossless and lossy compression for neural network weights: accuracy impact, compression ratio, and inference speed implications.",
    ]
    for p in comparison_topics:
        prompts.append({"prompt": p, "domain": "english", "ground_truth": None})

    # --- Category 5: Process/tutorial explanations (15 prompts) ---
    tutorial_topics = [
        "Write a step-by-step guide to deploying a smart contract on Ethereum: environment setup, writing the contract, testing with Hardhat, deploying to testnet, and verifying on Etherscan.",
        "Explain the process of training a language model from scratch: data collection, tokenizer training, architecture selection, training loop implementation, evaluation, and deployment.",
        "Write a guide to implementing a simple ZK circuit: defining the computation, creating constraints, generating a trusted setup, proving, and verifying.",
        "Explain the process of conducting a smart contract audit: methodology, common vulnerability patterns, tools (Slither, Mythril), reporting, and remediation verification.",
        "Write a guide to building a simple decentralized application (dApp): smart contract backend, frontend integration with ethers.js, MetaMask connection, and transaction handling.",
        "Explain the process of fine-tuning a language model with RLHF: training a reward model, implementing PPO, evaluation, and common pitfalls to avoid.",
        "Write a guide to implementing a Merkle tree in production: leaf hashing, tree construction, proof generation, proof verification, and handling tree updates.",
        "Explain the process of tokenizer training for a domain-specific language model: corpus preparation, BPE algorithm, vocabulary size selection, and evaluation.",
        "Write a guide to optimizing a neural network for edge deployment: profiling, quantization, pruning, operator fusion, and benchmarking on target hardware.",
        "Explain the process of building a blockchain indexer: listening for events, decoding transaction data, storing in a database, and serving queries via an API.",
        "Write a guide to implementing continuous integration for smart contracts: automated testing, gas reporting, security scanning, and deployment pipelines.",
        "Explain the process of setting up a validator node for a Proof of Stake network: hardware requirements, staking, key management, monitoring, and slashing prevention.",
        "Write a guide to implementing a simple state machine in Solidity: defining states and transitions, access control, event emission, and testing.",
        "Explain the process of data preparation for SFT: selecting domains, generating prompts, teacher model completions, verification, and filtering for quality.",
        "Write a guide to profiling and optimizing GPU utilization for deep learning: identifying bottlenecks, mixed precision training, gradient accumulation, and data loading.",
    ]
    for p in tutorial_topics:
        prompts.append({"prompt": p, "domain": "english", "ground_truth": None})

    return prompts[:150]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate hard SFT prompts")
    parser.add_argument("--output", type=str, default="data/sft_prompts_hard.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Only generate for specific domains")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print statistics, don't write")
    args = parser.parse_args()

    generators = {
        "math": generate_math_prompts,
        "sql": generate_sql_prompts,
        "solidity": generate_solidity_prompts,
        "sec_finance": generate_sec_finance_prompts,
        "english": generate_english_prompts,
    }

    all_prompts = []
    for domain, gen_fn in generators.items():
        if args.domains and domain not in args.domains:
            continue
        domain_prompts = gen_fn()
        all_prompts.extend(domain_prompts)
        print(f"  {domain}: {len(domain_prompts)} prompts")

    print(f"\nTotal: {len(all_prompts)} prompts")

    # Domain distribution
    from collections import Counter
    dist = Counter(p["domain"] for p in all_prompts)
    print("\nDomain distribution:")
    for d, c in sorted(dist.items(), key=lambda x: -x[1]):
        has_gt = sum(1 for p in all_prompts if p["domain"] == d and p["ground_truth"] is not None)
        print(f"  {d}: {c} ({has_gt} with ground truth)")

    if args.stats_only:
        return

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
