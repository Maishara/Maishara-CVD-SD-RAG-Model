import ast
from typing import List, Dict, Any, Tuple, Optional

class ASTVerificationError(Exception):
    """Custom exception raised when generated PySD code fails AST static analysis."""
    pass

class PySDASTValidator(ast.NodeVisitor):
    """
    AST Visitor that traverses and validates LLM-generated PySD Python code nodes.
    Enforces structural completeness, variable definitions, and execution safety.
    """
    def __init__(self, required_variables: Optional[List[str]] = None):
        self.required_variables = set(required_variables or [
            "initial_susceptible", 
            "initial_cvd", 
            "effective_incidence_rate", 
            "dt"
        ])
        self.defined_variables = set()
        self.forbidden_imports = {"os", "sys", "subprocess", "shutil", "builtins", "requests"}
        self.found_imports = []
        self.negative_assignments: List[Tuple[str, float]] = []

    def visit_Import(self, node: ast.Import):
        """Disallow forbidden library imports."""
        for alias in node.names:
            if alias.name in self.forbidden_imports:
                self.found_imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Disallow forbidden module imports."""
        if node.module in self.forbidden_imports:
            self.found_imports.append(node.module)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Track assigned variable names and check for negative constant values."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                self.defined_variables.add(var_name)

                # Check if a constant number is assigned a negative value where illegal
                if isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                    if isinstance(node.value.operand, (ast.Constant, ast.Num)):
                        val = -node.value.operand.value if hasattr(node.value.operand, 'value') else -node.value.operand.n
                        if var_name in {"initial_susceptible", "initial_cvd", "effective_incidence_rate"}:
                            self.negative_assignments.append((var_name, val))
                            
        self.generic_visit(node)

class ASTVerificationEngine:
    """
    Static verification engine executing AST analysis over raw code strings.
    """
    @staticmethod
    def verify_pysd_code(code_string: str, required_vars: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parses and validates a raw Python code string.
        Returns a verification report dict or raises ASTVerificationError.
        """
        # 1. Check basic Python syntax
        try:
            parsed_ast = ast.parse(code_string)
        except SyntaxError as e:
            raise ASTVerificationError(f"Syntax Error in generated code at line {e.lineno}: {e.msg}")

        # 2. Traverse AST nodes using validator
        validator = PySDASTValidator(required_variables=required_vars)
        validator.visit(parsed_ast)

        # 3. Security check: Check for forbidden imports
        if validator.found_imports:
            raise ASTVerificationError(
                f"Security Violation: Forbidden modules imported in generated code: {validator.found_imports}"
            )

        # 4. Check for missing required variables
        missing_vars = validator.required_variables - validator.defined_variables
        if missing_vars:
            raise ASTVerificationError(
                f"Structural Failure: Missing required System Dynamics variable assignments: {list(missing_vars)}"
            )

        # 5. Check for illegal negative rates/stocks
        if validator.negative_assignments:
            raise ASTVerificationError(
                f"Mathematical Error: Illegal negative values assigned to variables: {validator.negative_assignments}"
            )

        return {
            "status": "PASSED",
            "defined_variables": list(validator.defined_variables),
            "ast_nodes_count": len(list(ast.walk(parsed_ast)))
        }

# Test Verification Cases
if __name__ == "__main__":
    engine = ASTVerificationEngine()

    print("=== TEST 1: VALID PYSD CODE ===")
    valid_code = """
import numpy as np

dt = 0.1
initial_susceptible = 93200.0
initial_cvd = 6800.0
effective_incidence_rate = 0.012

# Rate equation
dS = - initial_susceptible * effective_incidence_rate * dt
"""
    report = engine.verify_pysd_code(valid_code)
    print("Result:", report)

    print("\n=== TEST 2: INVALID CODE (MISSING VARIABLE) ===")
    invalid_code_missing_var = """
dt = 0.1
initial_susceptible = 93200.0
# Missing initial_cvd and effective_incidence_rate
"""
    try:
        engine.verify_pysd_code(invalid_code_missing_var)
    except ASTVerificationError as e:
        print("Caught Expected Failure:", e)

    print("\n=== TEST 3: SECURITY VIOLATION (UNSAFE IMPORT) ===")
    unsafe_code = """
import os
os.system("rm -rf /")
dt = 0.1
initial_susceptible = 93200.0
initial_cvd = 6800.0
effective_incidence_rate = 0.012
"""
    try:
        engine.verify_pysd_code(unsafe_code)
    except ASTVerificationError as e:
        print("Caught Expected Failure:", e)