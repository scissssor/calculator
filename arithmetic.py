import random
import argparse
import sys
from fractions import Fraction
import re
from typing import List, Tuple, Optional, Set


class FractionNumber:
    """处理分数和带分数的类"""

    def __init__(self, numerator: int, denominator: int = 1):
        self.fraction = Fraction(numerator, denominator)

    @classmethod
    def from_string(cls, s: str) -> 'FractionNumber':
        """从字符串解析分数或带分数"""
        if "'" in s:
            # 带分数格式：a'b/c
            integer_part, fraction_part = s.split("'")
            numerator, denominator = map(int, fraction_part.split('/'))
            value = int(integer_part) + Fraction(numerator, denominator)
            return cls(value.numerator, value.denominator)
        elif '/' in s:
            # 真分数格式：a/b
            numerator, denominator = map(int, s.split('/'))
            return cls(numerator, denominator)
        else:
            # 自然数
            return cls(int(s))

    def to_display_string(self) -> str:
        """优化分数显示转换"""
        num, den = self.fraction.numerator, self.fraction.denominator

        if den == 1:
            return str(num)

        integer_part = num // den
        numerator = num % den

        if integer_part == 0:
            return f"{numerator}/{den}"
        else:
            return f"{integer_part}'{numerator}/{den}"

    # 添加缓存机制
    from functools import lru_cache

    @lru_cache(maxsize=1000)
    def cached_fraction(numerator: int, denominator: int = 1):
        return Fraction(numerator, denominator)


    def __add__(self, other):
        result = self.fraction + other.fraction
        return FractionNumber(result.numerator, result.denominator)

    def __sub__(self, other):
        result = self.fraction - other.fraction
        return FractionNumber(result.numerator, result.denominator)

    def __mul__(self, other):
        result = self.fraction * other.fraction
        return FractionNumber(result.numerator, result.denominator)

    def __truediv__(self, other):
        result = self.fraction / other.fraction
        return FractionNumber(result.numerator, result.denominator)

    def __lt__(self, other):
        return self.fraction < other.fraction

    def __le__(self, other):
        return self.fraction <= other.fraction

    def __eq__(self, other):
        return self.fraction == other.fraction

    def __str__(self):
        return self.to_display_string()


class ExpressionNode:
    """表达式树的节点"""

    def __init__(self, value=None, operator=None, left=None, right=None):
        self.value = value  # 对于叶子节点
        self.operator = operator  # 对于操作符节点
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.value is not None

    def evaluate(self) -> FractionNumber:
        """计算表达式的值"""
        if self.is_leaf():
            return self.value

        left_val = self.left.evaluate()
        right_val = self.right.evaluate()

        if self.operator == '+':
            return left_val + right_val
        elif self.operator == '-':
            # 确保不产生负数
            if left_val < right_val:
                raise ValueError("Subtraction result cannot be negative")
            return left_val - right_val
        elif self.operator == '×':
            return left_val * right_val
        elif self.operator == '÷':
            # 确保结果是真分数
            if left_val >= right_val:
                raise ValueError("Division result must be a proper fraction")
            return left_val / right_val

    def to_string(self, parent_precedence=0) -> str:
        """将表达式转换为字符串，处理括号"""
        if self.is_leaf():
            return str(self.value)

        # 定义运算符优先级
        precedence = {'+': 1, '-': 1, '×': 2, '÷': 2}

        left_str = self.left.to_string(precedence[self.operator])
        right_str = self.right.to_string(precedence[self.operator])

        # 处理减法和除法的特殊情况
        if self.operator == '-' and self.right.operator in ['+', '-']:
            right_str = f"({right_str})"
        elif self.operator == '÷' and self.right.operator in ['+', '-', '×', '÷']:
            right_str = f"({right_str})"

        result = f"{left_str} {self.operator} {right_str}"

        # 如果当前运算符优先级低于父运算符，需要加括号
        if parent_precedence > precedence[self.operator]:
            result = f"({result})"

        return result

    def normalize_form(self) -> str:
        """生成规范形式用于重复检测"""
        if self.is_leaf():
            return str(self.value)

        left_norm = self.left.normalize_form()
        right_norm = self.right.normalize_form()

        # 对于加法和乘法，排序左右子树以实现规范化
        if self.operator in ['+', '×']:
            if left_norm > right_norm:
                left_norm, right_norm = right_norm, left_norm

        return f"({left_norm}{self.operator}{right_norm})"


class ProblemGenerator:
    """题目生成器"""

    def __init__(self, range_limit: int):
        self.range_limit = range_limit
        self.generated_forms = set()

    def generate_number(self) -> FractionNumber:
        """生成一个数（自然数或真分数）"""
        if random.random() < 0.3:  # 30%概率生成分数
            denominator = random.randint(2, self.range_limit)
            numerator = random.randint(1, denominator - 1)
            return FractionNumber(numerator, denominator)
        else:
            return FractionNumber(random.randint(0, self.range_limit - 1))

    def generate_expression(self, max_operators: int) -> ExpressionNode:
        """递归生成表达式"""
        if max_operators == 0:
            return ExpressionNode(value=self.generate_number())

        # 修正运算符池的定义
        operators_pool = {
            1: ['+', '×', '-', '+', '×'],  # 1个运算符
            2: ['+', '×', '-', '+', '×'],  # 2个运算符
            3: ['+', '×', '-', '÷', '+', '×']  # 3个运算符
        }

        # 确保 max_operators 在有效范围内
        if max_operators not in operators_pool:
            max_operators = min(max_operators, max(operators_pool.keys()))

        operator = random.choice(operators_pool[max_operators])

        # 更灵活地分配运算符数量
        if max_operators == 1:
            left_operators = 0
            right_operators = 0
        else:
            left_operators = random.randint(0, max_operators - 1)
            right_operators = max_operators - 1 - left_operators

        left_expr = self.generate_expression(left_operators)
        right_expr = self.generate_expression(right_operators)

        # 对于减法和除法，确保条件满足
        if operator in ['-', '÷']:
            return self._validate_operation(operator, left_expr, right_expr, right_operators)

        return ExpressionNode(operator=operator, left=left_expr, right=right_expr)

    def _validate_operation(self, operator, left_expr, right_expr, right_operators):
        """提取验证逻辑，减少重复代码"""
        left_val = left_expr.evaluate()
        right_val = right_expr.evaluate()

        if operator == '-':
            if left_val < right_val:
                left_expr, right_expr = right_expr, left_expr
        elif operator == '÷':
            if left_val >= right_val:
                # 限制重试次数，避免无限循环
                for _ in range(5):
                    new_right = self.generate_expression(right_operators)
                    new_right_val = new_right.evaluate()
                    if new_right_val > left_val and new_right_val != FractionNumber(0):
                        right_expr = new_right
                        break
                else:
                    operator = '×'  # 快速回退

        return ExpressionNode(operator=operator, left=left_expr, right=right_expr)

    def generate_unique_problem(self, max_attempts=1000) -> Tuple[str, str]:
        """生成不重复的题目"""
        for _ in range(max_attempts):
            try:
                operators_count = random.randint(1, 3)
                expression = self.generate_expression(operators_count)

                # 检查计算过程是否有效
                result = expression.evaluate()

                # 检查是否重复
                normalized_form = expression.normalize_form()
                if normalized_form in self.generated_forms:
                    continue

                self.generated_forms.add(normalized_form)
                problem_str = f"{expression.to_string()} ="
                answer_str = str(result)

                return problem_str, answer_str

            except (ValueError, ZeroDivisionError):
                continue

        raise Exception("Failed to generate unique problem after maximum attempts")


    def generate_problems(self, count: int) -> List[Tuple[str, str]]:
        """批量生成题目，减少中间状态保存"""
        problems = []
        batch_size = 100  # 分批处理

        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_problems = []

            for i in range(batch_start, batch_end):
                problem, answer = self.generate_unique_problem()
                batch_problems.append((problem, answer))

            problems.extend(batch_problems)

            # 每批完成后清理临时缓存
            if hasattr(self, '_temp_cache'):
                self._temp_cache.clear()

        return problems

class Grader:
    """题目批改器"""

    @staticmethod
    def evaluate_expression(expression: str) -> FractionNumber:
        """计算表达式的值"""
        # 移除空格
        expression = expression.replace(' ', '')

        def parse_expression(tokens):
            """递归解析表达式"""

            def parse_term(tokens):
                if not tokens:
                    raise ValueError("Unexpected end of expression")

                token = tokens.pop(0)
                if token == '(':
                    node = parse_expression(tokens)
                    if tokens.pop(0) != ')':
                        raise ValueError("Missing closing parenthesis")
                    return node
                elif token.replace('/', '').isdigit() or "'" in token:
                    return ExpressionNode(value=FractionNumber.from_string(token))
                else:
                    raise ValueError(f"Unexpected token: {token}")

            node = parse_term(tokens)

            while tokens and tokens[0] in ['+', '-', '×', '÷']:
                operator = tokens.pop(0)
                right = parse_term(tokens)
                node = ExpressionNode(operator=operator, left=node, right=right)

            return node

        # 简单的分词器
        tokens = []
        i = 0
        while i < len(expression):
            if expression[i] in '()+×÷-':
                tokens.append(expression[i])
                i += 1
            else:
                # 读取数字或分数
                start = i
                while i < len(expression) and (expression[i].isdigit() or expression[i] in '/\''):
                    i += 1
                tokens.append(expression[start:i])

        expression_tree = parse_expression(tokens)
        return expression_tree.evaluate()

    def grade(self, exercise_file: str, answer_file: str) -> Tuple[List[int], List[int]]:
        """批改题目"""
        correct_indices = []
        wrong_indices = []

        with open(exercise_file, 'r', encoding='utf-8') as ex_file, \
                open(answer_file, 'r', encoding='utf-8') as ans_file:

            exercises = ex_file.readlines()
            answers = ans_file.readlines()

            for i, (exercise, answer) in enumerate(zip(exercises, answers)):
                try:
                    # 提取表达式部分（去掉等号）
                    expr_str = exercise.strip().rstrip('= ')
                    expected = FractionNumber.from_string(answer.strip())
                    actual = self.evaluate_expression(expr_str)

                    if actual == expected:
                        correct_indices.append(i + 1)  # 题目编号从1开始
                    else:
                        wrong_indices.append(i + 1)
                except Exception:
                    wrong_indices.append(i + 1)

        return correct_indices, wrong_indices


def main():


    parser = argparse.ArgumentParser(description='小学四则运算题目生成器')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', type=int, help='生成题目的数量')
    group.add_argument('-e', type=str, help='题目文件路径')
    parser.add_argument('-r', type=int, help='数值范围')
    parser.add_argument('-a', type=str, help='答案文件路径')

    args = parser.parse_args()

    if args.n is not None:
        # 生成模式
        if args.r is None:
            print("错误：生成模式必须使用 -r 参数指定数值范围")
            sys.exit(1)

        print(f"生成 {args.n} 道题目，数值范围: {args.r}")
        generator = ProblemGenerator(args.r)
        problems = generator.generate_problems(args.n)

        # 写入题目文件
        with open('Exercises.txt', 'w', encoding='utf-8') as ex_file:
            for problem, _ in problems:
                ex_file.write(problem + '\n')

        # 写入答案文件
        with open('Answers.txt', 'w', encoding='utf-8') as ans_file:
            for _, answer in problems:
                ans_file.write(answer + '\n')

        print(f"题目已保存到 Exercises.txt")
        print(f"答案已保存到 Answers.txt")

    else:
        # 批改模式
        if args.e is None or args.a is None:
            print("错误：批改模式必须使用 -e 和 -a 参数")
            sys.exit(1)

        print(f"批改题目文件: {args.e}")
        print(f"答案文件: {args.a}")

        grader = Grader()
        correct, wrong = grader.grade(args.e, args.a)

        # 写入批改结果
        with open('Grade.txt', 'w', encoding='utf-8') as grade_file:
            grade_file.write(f"Correct: {len(correct)} ({', '.join(map(str, correct))})\n")
            grade_file.write(f"Wrong: {len(wrong)} ({', '.join(map(str, wrong))})\n")

        print(f"批改结果已保存到 Grade.txt")
        print(f"正确: {len(correct)} 题")
        print(f"错误: {len(wrong)} 题")


if __name__ == '__main__':
    main()