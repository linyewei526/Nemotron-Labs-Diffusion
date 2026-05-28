#!/usr/bin/env python3
"""
Patch for NeMo-Skills to fix DictConfig JSON serialization errors.

When Hydra creates nested config objects (e.g. extra_body.chat_template_kwargs),
they become OmegaConf DictConfig objects. These are not JSON-serializable and
cause litellm/httpx to fail with:
    TypeError: Object of type DictConfig is not JSON serializable

This patch wraps the litellm.acompletion call in base.py to recursively convert
any DictConfig/ListConfig values in request_params to plain Python dicts/lists.
"""

import os
import sys
import re
import shutil
import importlib.util


def find_base_model_path():
    """Find nemo_skills/inference/model/base.py."""
    try:
        spec = importlib.util.find_spec("nemo_skills")
        if spec is None or spec.origin is None:
            return None
        package_dir = os.path.dirname(spec.origin)
        base_path = os.path.join(package_dir, "inference", "model", "base.py")
        if os.path.exists(base_path):
            return base_path
        return None
    except (ImportError, AttributeError):
        return None


RESOLVE_HELPER = '''
# PATCH: Resolve OmegaConf DictConfig/ListConfig to plain Python types
# before passing to litellm, which requires JSON-serializable dicts.
def _resolve_omegaconf(obj):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _resolve_omegaconf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_omegaconf(v) for v in obj]
    return obj
'''

PATCH_MARKER = "# PATCH: Resolve OmegaConf DictConfig/ListConfig"


def patch_base_model():
    """Patch base.py to resolve DictConfig before litellm calls."""
    base_path = find_base_model_path()
    if base_path is None:
        print("Could not locate nemo_skills base.py")
        return False

    print(f"Patching: {base_path}")

    with open(base_path, 'r') as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("Already patched!")
        return True

    backup_path = base_path + ".backup_dictconfig"
    if not os.path.exists(backup_path):
        shutil.copy2(base_path, backup_path)
        print(f"Backup: {backup_path}")

    # Insert the helper function near the top, after the last top-level import
    import_block_end = 0
    for match in re.finditer(r'^(?:import |from )\S+', content, re.MULTILINE):
        line_end = content.index('\n', match.start())
        if line_end > import_block_end:
            import_block_end = line_end

    if import_block_end == 0:
        print("Could not find import block in base.py")
        return False

    content = content[:import_block_end + 1] + '\n' + RESOLVE_HELPER + content[import_block_end + 1:]

    # Find the litellm.acompletion call, preserving its exact indentation.
    # The line may be inside a try: block, so we must keep the indent level.
    pattern = re.compile(
        r'^( +)(response\s*=\s*await\s+litellm\.acompletion\(\*\*request_params)',
        re.MULTILINE,
    )
    match = pattern.search(content)
    if match:
        indent = match.group(1)
        original_line_start = match.group(0)
        replacement = f'{indent}request_params = _resolve_omegaconf(request_params)\n{original_line_start}'
        content = content.replace(original_line_start, replacement, 1)
    else:
        print("WARNING: Could not find litellm.acompletion call to patch.")
        print("You may need to manually add: request_params = _resolve_omegaconf(request_params)")
        print("before the litellm.acompletion() call in base.py")

    with open(base_path, 'w') as f:
        f.write(content)

    # Verify no syntax errors
    try:
        with open(base_path, 'r') as f:
            compile(f.read(), base_path, 'exec')
        print("Patch applied successfully! (syntax verified)")
    except SyntaxError as e:
        print(f"WARNING: Patch introduced a syntax error at line {e.lineno}: {e.msg}")
        print("Restoring from backup...")
        shutil.copy2(backup_path, base_path)
        print("Backup restored. Patch was NOT applied.")
        return False

    return True


if __name__ == "__main__":
    print("NeMo-Skills DictConfig Serialization Fix")
    print("=" * 50)
    if patch_base_model():
        print("\nDictConfig values (e.g. chat_template_kwargs) will be auto-resolved.")
    else:
        print("\nPatch failed.")
        sys.exit(1)
