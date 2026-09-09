"""Grade saved code functions in a restricted, resource-limited child process."""

import argparse
import bisect
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

FUNCTIONS = {
    "coding_intervals": "merge_intervals",
    "coding_brackets": "balanced",
    "coding_search": "lower_bound",
    "coding_unique": "first_unique",
}


def extract_final_answer(text):
    """Conservative extraction from an explicit final line, never intermediate work."""
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    tail = re.sub(r"^(?:Final answer|Answer)\s*:\s*", "", tail, flags=re.I)
    return tail.strip(" *_$")


def cases(name):
    rng = random.Random(21)
    if name == "coding_search":
        data = [
            (sorted(rng.choices(range(-20, 21), k=rng.randrange(30))), rng.randrange(-25, 26))
            for _ in range(100)
        ]
        return [([v, t], bisect.bisect_left(v, t)) for v, t in data]
    if name == "coding_unique":
        texts = ["", "a", "aabb", "swiss"] + [
            "".join(rng.choices("abcdxyz", k=rng.randrange(40))) for _ in range(100)
        ]
        return [([t], next((c for c in t if t.count(c) == 1), None)) for t in texts]
    if name == "coding_brackets":

        def expected(text):
            stack = []
            closing = {")": "(", "]": "[", "}": "{"}
            for c in text:
                if c in "([{":
                    stack.append(c)
                elif c in closing:
                    if not stack or stack.pop() != closing[c]:
                        return False
            return not stack

        texts = ["", "abc", "([]{})", "([)]", "a(b[c]d)e", "(("] + [
            "".join(rng.choices("()[]{}abc", k=rng.randrange(30))) for _ in range(100)
        ]
        return [([t], expected(t)) for t in texts]

    def merge(intervals):
        out = []
        for a, b in sorted(intervals):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    data = [[], [[1, 2], [2, 3]], [[5, 7], [1, 2]]] + [
        [sorted(rng.choices(range(30), k=2)) for _ in range(rng.randrange(20))] for _ in range(100)
    ]
    return [([v], merge(v)) for v in data]


WORKER = r"""
import ast,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
p=json.load(sys.stdin)
tree=ast.parse(p['code'])
bad=(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith,ast.Await,ast.Yield,ast.YieldFrom)
methods={'add','append','pop','get','setdefault','keys','values','items','copy','sort','extend','count','index'}
for node in ast.walk(tree):
    if isinstance(node,bad):raise ValueError('Unsupported construct')
    if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('Private name')
    if isinstance(node,ast.Attribute) and node.attr not in methods:raise ValueError('Unsupported attribute')
    if isinstance(node,ast.FunctionDef):
        if node.decorator_list:raise ValueError('Decorator')
        node.returns=None
        for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
for node in tree.body:
    if not isinstance(node,ast.FunctionDef) and not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
        raise ValueError('Top-level statement')
builtins={'len':len,'range':range,'enumerate':enumerate,'sorted':sorted,'list':list,'dict':dict,'set':set,'tuple':tuple,'min':min,'max':max,'sum':sum,'abs':abs,'int':int,'bool':bool,'str':str,'zip':zip,'reversed':reversed,'any':any,'all':all,'ValueError':ValueError}
scope={'__builtins__':builtins}
exec(compile(ast.fix_missing_locations(tree),'<generated function>','exec'),scope)
function=scope[p['function']]
passed=0;failures=[]
for args,expected in p['cases']:
    actual=function(*args)
    # Interval containers are unspecified; compare endpoint sequences, not tuple/list types.
    if p['function']=='merge_intervals':actual=[list(interval) for interval in actual]
    if actual==expected:passed+=1
    elif len(failures)<3:failures.append({'input':args,'expected':expected,'actual':actual})
print(json.dumps({'passed':passed,'total':len(p['cases']),'failures':failures}))
"""


def grade(path):
    report = json.loads(path.read_text())
    for task in report["tasks"]:
        if task["expected"] is not None:
            # Only an explicit final line counts; never search earlier reasoning text.
            tail = extract_final_answer(task["text"])
            task["final_answer_extracted"] = tail
            task["semantic_answer_match"] = tail == task["expected"]
        if task["name"] not in FUNCTIONS:
            continue
        match = re.search(r"```(?:python)?\s*\n(.*?)```", task["text"], re.S)
        code = match.group(1) if match else task["text"]
        payload = {"code": code, "function": FUNCTIONS[task["name"]], "cases": cases(task["name"])}
        try:
            run = subprocess.Popen(
                [sys.executable, "-B", "-I", "-S", "-c", WORKER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            run.stdin.write(json.dumps(payload))
            run.stdin.close()
            start = time.monotonic()
            reason = None
            process = psutil.Process(run.pid)
            while run.poll() is None:
                try:
                    rss = process.memory_info().rss
                except psutil.NoSuchProcess:
                    break
                if rss > 512 * 1024 * 1024 or time.monotonic() - start > 4:
                    reason = "resource limit"
                    run.kill()
                    break
                time.sleep(0.01)
            run.wait()
            stdout = run.stdout.read(65536)
            stderr = run.stderr.read(4096)
            result = (
                json.loads(stdout)
                if run.returncode == 0
                else {"error": reason or (stderr.splitlines()[-1] if stderr else "worker failed")}
            )
        except subprocess.TimeoutExpired:
            result = {"error": "resource timeout"}
        task["functional_grade"] = result
    report["grading"] = (
        "strict full-answer equality and separate final-line answer extraction; constrained functional tests for four code functions; interval tuple/list containers normalized because prompt does not specify them; all outputs and token IDs retained"
    )
    report["timing_scope"] = (
        "single-pass workload timings, not shape-warmed controlled speed benchmarks"
    )
    path.write_text(json.dumps(report, indent=2))
    print(
        path.name,
        [
            (t["name"], t.get("functional_grade"))
            for t in report["tasks"]
            if "functional_grade" in t
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    for path in parser.parse_args().reports:
        grade(path)
