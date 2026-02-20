#!/usr/bin/env python3
"""
Test script to verify visualize_vit_attention.py has no import syntax errors.
"""
import ast
import os
import sys

def check_imports(filename):
    """Parse Python file and verify imports are syntactically valid."""
    try:
        with open(filename, 'r') as f:
            code = f.read()

        # Parse the file to check for syntax errors
        tree = ast.parse(code, filename=filename)

        # Extract all imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        print(f"✓ {filename} has valid Python syntax")
        print(f"✓ Found {len(imports)} import statements")
        print("✓ Imports: ", ', '.join(sorted(set(imports))))
        return True

    except SyntaxError as e:
        print(f"✗ Syntax error in {filename}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filename}: {e}")
        return False

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "visualize_vit_attention.py")
    success = check_imports(target)
    sys.exit(0 if success else 1)
