#!/usr/bin/env python3
"""Patch NeMo-Skills IFEval scorer path lookup.

The upstream NeMo-Skills evaluator hard-codes
``/opt/benchmarks/google-research``. That works in their container images but
breaks in the local SGLang pipeline. This patch keeps the original fallback and
adds support for ``NLD_GOOGLE_RESEARCH_DIR`` / ``GOOGLE_RESEARCH_DIR``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys


PATCH_MARKER = "# PATCH: configurable Google Research path for IFEval"


def find_ifeval_evaluator_path() -> str | None:
    spec = importlib.util.find_spec("nemo_skills")
    if spec is None or spec.origin is None:
        return None
    package_dir = os.path.dirname(spec.origin)
    path = os.path.join(package_dir, "evaluation", "evaluator", "ifeval.py")
    return path if os.path.exists(path) else None


def patch_ifeval_evaluator() -> bool:
    path = find_ifeval_evaluator_path()
    if path is None:
        print("Could not locate nemo_skills/evaluation/evaluator/ifeval.py")
        return False

    print(f"Patching: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if PATCH_MARKER in content:
        changed = False
        if "import sys\n" not in content:
            content = content.replace("import subprocess\n", "import subprocess\nimport sys\n", 1)
            changed = True
        legacy = (
            '        f"cd {shlex.quote(google_research_dir)} && python -m '
            'instruction_following_eval.evaluation_main "\n'
        )
        upgraded = (
            '        f"cd {shlex.quote(google_research_dir)} && {shlex.quote(sys.executable)} -m '
            'instruction_following_eval.evaluation_main "\n'
        )
        if legacy in content:
            content = content.replace(legacy, upgraded, 1)
            changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            compile(content, path, "exec")
            print("Existing patch upgraded successfully!")
        else:
            print("Already patched!")
        return True

    backup_path = path + ".backup_ifeval_google_research_path"
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f"Backup: {backup_path}")

    if "import os\n" not in content:
        content = content.replace("import logging\n", "import logging\nimport os\n", 1)
    if "import shlex\n" not in content:
        content = content.replace("import shutil\n", "import shutil\nimport shlex\n", 1)
    if "import sys\n" not in content:
        content = content.replace("import subprocess\n", "import subprocess\nimport sys\n", 1)

    old = (
        '    cmd = (\n'
        '        "cd /opt/benchmarks/google-research && python -m instruction_following_eval.evaluation_main "\n'
        '        f"--input_data={jsonl_file} "\n'
        '        f"--input_response_data={jsonl_file} "\n'
        '        f"--output_dir={output_dir} "\n'
        '    )\n'
    )
    new = (
        f"    {PATCH_MARKER}\n"
        '    google_research_dir = os.environ.get("NLD_GOOGLE_RESEARCH_DIR") or os.environ.get(\n'
        '        "GOOGLE_RESEARCH_DIR", "/opt/benchmarks/google-research"\n'
        "    )\n"
        "    if not Path(google_research_dir).is_dir():\n"
        "        raise FileNotFoundError(\n"
        '            f"Google Research IFEval scorer directory not found: {google_research_dir}. "\n'
        '            "Set NLD_GOOGLE_RESEARCH_DIR to a google-research checkout containing "\n'
        '            "instruction_following_eval."\n'
        "        )\n"
        "    cmd = (\n"
        '        f"cd {shlex.quote(google_research_dir)} && {shlex.quote(sys.executable)} -m instruction_following_eval.evaluation_main "\n'
        '        f"--input_data={shlex.quote(str(jsonl_file))} "\n'
        '        f"--input_response_data={shlex.quote(str(jsonl_file))} "\n'
        '        f"--output_dir={shlex.quote(str(output_dir))} "\n'
        "    )\n"
    )

    if old not in content:
        print("Could not find the expected hard-coded IFEval command block.")
        return False
    content = content.replace(old, new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        compile(content, path, "exec")
    except SyntaxError as exc:
        print(f"Patch introduced a syntax error at line {exc.lineno}: {exc.msg}")
        shutil.copy2(backup_path, path)
        return False

    print("Patch applied successfully! (syntax verified)")
    return True


if __name__ == "__main__":
    print("NeMo-Skills IFEval Google Research Path Fix")
    print("=" * 50)
    if patch_ifeval_evaluator():
        print("\nIFEval will use NLD_GOOGLE_RESEARCH_DIR / GOOGLE_RESEARCH_DIR when set.")
    else:
        print("\nPatch failed.")
        sys.exit(1)
