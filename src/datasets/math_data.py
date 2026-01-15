import numpy as np
import math
import sympy
import re
from typing import List, Dict, Tuple, Optional, Any
from fractions import Fraction

REASONING_TEMPLATES = [
    "Calculate {a} + {b}",
    "Calculate {a} - {b}",
    "Calculate {a} * {b}",
    "Calculate {a} / {b}",
    "Calculate {percent}% of {value}",
    "Let {var} = {expr}",
    "Substitute {var} = {value}",
    "The total is {a} + {b}",
    "The difference is {a} - {b}",
    "Multiply {a} by {b}",
    "Divide {a} by {b}",
    "{a} times {b} equals {result}",
    "Add {a} and {b} to get {result}",
    "Subtract {b} from {a} to get {result}",
]


class MathDataset:

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.problems = []

    def verify_step(self, step_text: str) -> Tuple[bool, Optional[Any]]:
        try:
            patterns = [
                r'(\d+(?:\.\d+)?)\s*[\+]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*[\-]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*[\*×x]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*[\/÷]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            ]

            for pattern in patterns:
                match = re.search(pattern, step_text)
                if match:
                    a, b, expected = map(float, match.groups())

                    if '+' in step_text:
                        result = a + b
                    elif '-' in step_text:
                        result = a - b
                    elif any(c in step_text for c in '*×x'):
                        result = a * b
                    elif any(c in step_text for c in '/÷'):
                        result = a / b if b != 0 else float('inf')
                    else:
                        continue

                    valid = abs(result - expected) < 0.01
                    return valid, result

            calc_pattern = r'[Cc]alculate\s+(\d+(?:\.\d+)?)\s*([\+\-\*\/×÷])\s*(\d+(?:\.\d+)?)'
            match = re.search(calc_pattern, step_text)
            if match:
                a, op, b = match.groups()
                a, b = float(a), float(b)

                if op in ['+']:
                    result = a + b
                elif op in ['-']:
                    result = a - b
                elif op in ['*', '×']:
                    result = a * b
                elif op in ['/', '÷']:
                    result = a / b if b != 0 else float('inf')
                else:
                    return False, None

                return True, result

            return True, None

        except Exception as e:
            return False, None

    def get_step_reward(self, step_text: str) -> float:
        valid, _ = self.verify_step(step_text)
        return 1.0 if valid else -0.5

    def check_final_answer(self,
                           predicted: float,
                           ground_truth: float,
                           tolerance: float = 0.01) -> bool:
        try:
            return abs(float(predicted) - float(ground_truth)) < tolerance
        except:
            return str(predicted).strip() == str(ground_truth).strip()


class GSM8KDataset(MathDataset):

    def __init__(self, n_problems: int = 100, seed: int = 42):
        super().__init__(seed)
        self.n_problems = n_problems
        self.problems = self._generate_synthetic_problems()

    def _generate_synthetic_problems(self) -> List[Dict]:
        problems = []
        templates = [
            {
                "template":
                "{name} has {n1} apples. {name2} gives {name} {n2} more apples. How many apples does {name} have now?",
                "solution_template": "{n1} + {n2} = {answer}",
                "compute": lambda n1, n2: n1 + n2
            },
            {
                "template":
                "{name} buys {n1} items at ${n2} each. What is the total cost?",
                "solution_template": "{n1} × {n2} = {answer}",
                "compute": lambda n1, n2: n1 * n2
            },
            {
                "template":
                "{name} has ${n1}. {name} spends ${n2}. How much money does {name} have left?",
                "solution_template": "{n1} - {n2} = {answer}",
                "compute": lambda n1, n2: n1 - n2
            },
            {
                "template":
                "{name} wants to split {n1} cookies equally among {n2} friends. How many cookies does each friend get?",
                "solution_template": "{n1} ÷ {n2} = {answer}",
                "compute": lambda n1, n2: n1 // n2 if n2 != 0 else 0
            },
            {
                "template":
                "{name} reads {n1} pages per day. How many pages will {name} read in {n2} days?",
                "solution_template": "{n1} × {n2} = {answer}",
                "compute": lambda n1, n2: n1 * n2
            },
            {
                "template":
                "A store has {n1} shirts. They sell {n2} shirts. How many shirts are left?",
                "solution_template": "{n1} - {n2} = {answer}",
                "compute": lambda n1, n2: n1 - n2
            },
            {
                "template":
                "{name} has {n1} stickers. {name} gets {n2} more stickers from {name2}. How many stickers does {name} have in total?",
                "solution_template": "{n1} + {n2} = {answer}",
                "compute": lambda n1, n2: n1 + n2
            },
            {
                "template":
                "There are {n1} students in a class. Each student needs {n2} pencils. How many pencils are needed in total?",
                "solution_template": "{n1} × {n2} = {answer}",
                "compute": lambda n1, n2: n1 * n2
            },
        ]

        names = [
            "Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace",
            "Henry"
        ]

        for i in range(self.n_problems):
            template_data = templates[i % len(templates)]

            n1 = np.random.randint(5, 100)
            n2 = np.random.randint(2, min(50, n1))
            name = names[i % len(names)]
            name2 = names[(i + 1) % len(names)]

            answer = template_data["compute"](n1, n2)

            question = template_data["template"].format(name=name,
                                                        name2=name2,
                                                        n1=n1,
                                                        n2=n2)

            solution = template_data["solution_template"].format(n1=n1,
                                                                 n2=n2,
                                                                 answer=answer)

            problems.append({
                "id": i,
                "question": question,
                "answer": answer,
                "solution": solution,
                "numbers": [n1, n2],
                "difficulty": "easy"
            })

        return problems

    def generate_step_candidates(self,
                                 problem: Dict,
                                 current_state: Dict,
                                 n_candidates: int = 50) -> List[str]:
        numbers = problem["numbers"]
        candidates = []

        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]

            candidates.extend([
                f"Calculate {a} + {b} = {a + b}",
                f"Calculate {a} - {b} = {a - b}",
                f"Calculate {a} * {b} = {a * b}",
                f"Calculate {a} / {b} = {a / b:.2f}"
                if b != 0 else f"Calculate {a} / {b} = undefined",
                f"Add {a} and {b} to get {a + b}",
                f"Subtract {b} from {a} to get {a - b}",
                f"Multiply {a} by {b} to get {a * b}",
                f"{a} times {b} equals {a * b}",
                f"The sum of {a} and {b} is {a + b}",
                f"The difference between {a} and {b} is {a - b}",
            ])

        while len(candidates) < n_candidates:
            rand_a = np.random.randint(1, 100)
            rand_b = np.random.randint(1, 100)
            op = np.random.choice(['+', '-', '*', '/'])

            if op == '+':
                result = rand_a + rand_b
            elif op == '-':
                result = rand_a - rand_b
            elif op == '*':
                result = rand_a * rand_b
            else:
                result = rand_a / rand_b if rand_b != 0 else 0

            is_wrong = np.random.random() < 0.3
            if is_wrong:
                result = result + np.random.randint(-10, 10)

            candidates.append(
                f"Calculate {rand_a} {op} {rand_b} = {result:.2f}")

        np.random.shuffle(candidates)
        return candidates[:n_candidates]

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]


class MATH500Dataset(MathDataset):

    def __init__(self, n_problems: int = 50, seed: int = 42):
        super().__init__(seed)
        self.n_problems = n_problems
        self.problems = self._generate_harder_problems()

    def _generate_harder_problems(self) -> List[Dict]:
        problems = []

        difficulty_levels = ["Level 3", "Level 4", "Level 5"]
        subjects = [
            "Algebra", "Geometry", "Number Theory", "Probability", "Counting"
        ]

        for i in range(self.n_problems):
            subject = subjects[i % len(subjects)]
            difficulty = difficulty_levels[i % len(difficulty_levels)]

            if subject == "Algebra":
                a = np.random.randint(2, 10)
                b = np.random.randint(1, 20)
                c = np.random.randint(1, 20)
                answer = (c - b) / a if a != 0 else 0
                question = f"Solve for x: {a}x + {b} = {c}"
                solution = f"Subtract {b} from both sides: {a}x = {c - b}. Divide by {a}: x = {answer:.2f}"

            elif subject == "Geometry":
                r = np.random.randint(2, 15)
                answer = 3.14159 * r * r
                question = f"Find the area of a circle with radius {r}."
                solution = f"Area = π × r² = π × {r}² = {answer:.2f}"

            elif subject == "Number Theory":
                n = np.random.randint(10, 50)
                answer = sum(i for i in range(1, n + 1) if n % i == 0)
                question = f"Find the sum of all positive divisors of {n}."
                divisors = [i for i in range(1, n + 1) if n % i == 0]
                solution = f"Divisors of {n}: {divisors}. Sum = {answer}"

            elif subject == "Probability":
                n = np.random.randint(2, 6)
                k = np.random.randint(1, n)
                total = 2**n
                favorable = int(math.comb(n, k))
                answer = favorable / total
                question = f"A fair coin is flipped {n} times. What is the probability of getting exactly {k} heads?"
                solution = f"P = C({n},{k}) / 2^{n} = {favorable}/{total} = {answer:.4f}"

            else:
                n = np.random.randint(3, 8)
                answer = int(math.factorial(n))
                question = f"In how many ways can {n} people be arranged in a line?"
                solution = f"{n}! = {answer}"

            problems.append({
                "id": i,
                "question": question,
                "answer": answer,
                "solution": solution,
                "subject": subject,
                "difficulty": difficulty
            })

        return problems

    def generate_step_candidates(self,
                                 problem: Dict,
                                 current_state: Dict,
                                 n_candidates: int = 100) -> List[str]:
        candidates = []

        subject = problem.get("subject", "Algebra")

        if subject == "Algebra":
            for _ in range(n_candidates // 2):
                a = np.random.randint(1, 20)
                b = np.random.randint(1, 20)
                op = np.random.choice(['+', '-', '*', '/'])

                if op == '+':
                    result = a + b
                elif op == '-':
                    result = a - b
                elif op == '*':
                    result = a * b
                else:
                    result = a / b if b != 0 else 0

                candidates.append(f"Calculate {a} {op} {b} = {result:.2f}")

        while len(candidates) < n_candidates:
            step_types = [
                "Simplify the expression",
                "Apply the formula",
                "Substitute the value",
                "Rearrange the equation",
                "Factor the expression",
                "Expand the brackets",
            ]
            candidates.append(np.random.choice(step_types))

        np.random.shuffle(candidates)
        return candidates[:n_candidates]

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]
