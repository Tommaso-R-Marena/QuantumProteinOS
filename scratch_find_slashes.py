import os
import ast

def check_files():
    files = [os.path.join(r, f) for r, d, files in os.walk('.') for f in files if f.endswith('.py')]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # very simple search for backslash in string literals using AST to avoid comments
        try:
            tree = ast.parse(content)
        except Exception:
            continue
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if '\\' in node.value and not node.value.startswith('\\') and not node.value in ('\n', '\t', '\r'):
                    # might be an escape, but let's print all
                    print(f'{file}: {node.value}')

check_files()
