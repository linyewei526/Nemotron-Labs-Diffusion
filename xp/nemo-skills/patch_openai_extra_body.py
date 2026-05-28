#!/usr/bin/env python3
"""
Patch for NeMo-Skills OpenAI model to fix missing extra_body parameter.
This script patches the OpenAI model to properly include extra_body in requests.
"""

import os
import sys
import importlib.util

def find_nemo_skills_path():
    """Dynamically find the nemo_skills installation path."""
    try:
        spec = importlib.util.find_spec("nemo_skills")
        if spec is None or spec.origin is None:
            return None
        # Get the package directory
        package_dir = os.path.dirname(spec.origin)
        openai_path = os.path.join(package_dir, "inference", "model", "openai.py")
        if os.path.exists(openai_path):
            return openai_path
        return None
    except (ImportError, AttributeError):
        return None

def patch_openai_model():
    """Apply patch to fix extra_body parameter in OpenAI model."""
    
    nemo_skills_path = find_nemo_skills_path()
    
    if nemo_skills_path is None:
        print("❌ Could not locate nemo_skills installation. Is it installed?")
        return False
    
    print(f"🔧 Patching NeMo-Skills OpenAI model at: {nemo_skills_path}")
    print("   This fixes missing extra_body parameter (including generation-algorithm)...")
    
    # Read the current file
    with open(nemo_skills_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "# PATCH: Include extra_body parameter" in content:
        print("✅ Already patched!")
        return True
    
    # Apply the patch - add extra_body handling before return params
    original_return = "        return params"
    patched_return = """        # PATCH: Include extra_body parameter (missing in original NeMo-Skills)
        if extra_body:
            params["extra_body"] = extra_body
        
        return params"""
    
    if original_return not in content:
        print("❌ Could not find the return statement to patch!")
        return False
    
    # Replace the return statement
    patched_content = content.replace(original_return, patched_return)
    
    # Backup original file
    backup_path = nemo_skills_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"📄 Backup created: {backup_path}")
    
    # Write patched version
    with open(nemo_skills_path, 'w') as f:
        f.write(patched_content)
    
    print("✅ Patch applied successfully!")
    print("   The OpenAI model will now properly pass extra_body parameters.")
    
    return True

def test_patch():
    """Test that the patch works."""
    print("\n🧪 Testing the patch...")
    
    try:
        from nemo_skills.inference.model.openai import OpenAIModel
        
        # Create client and test
        client = OpenAIModel(
            model='nemotron-labs-diffusion-8b',
            base_url='http://localhost:8000/v1',
            api_key='test'
        )
        
        # Test parameter building
        params = client._build_chat_request_params(
            messages=[{'role': 'user', 'content': 'test'}],
            tokens_to_generate=64,
            temperature=0.7,
            top_p=0.95,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            random_seed=42,
            stop_phrases=None,
            timeout=None,
            top_logprobs=None,
            stream=False,
            reasoning_effort=None,
            extra_body={'steps': 128, 'cfg_scale': 2.0, 'remasking': 'random'},
            tools=None
        )
        
        if 'extra_body' in params:
            print("✅ PATCH SUCCESSFUL! extra_body is now included:")
            print(f"   extra_body: {params['extra_body']}")
            return True
        else:
            print("❌ PATCH FAILED! extra_body still missing.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def restore_backup():
    """Restore from backup if needed."""
    nemo_skills_path = find_nemo_skills_path()
    
    if nemo_skills_path is None:
        print("❌ Could not locate nemo_skills installation.")
        return False
    
    backup_path = nemo_skills_path + ".backup"
    
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            original_content = f.read()
        
        with open(nemo_skills_path, 'w') as f:
            f.write(original_content)
        
        print("🔄 Restored from backup")
        return True
    else:
        print("❌ No backup found")
        return False

if __name__ == "__main__":
    print("🚀 NeMo-Skills OpenAI Model Patcher")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_backup()
    else:
        if patch_openai_model():
            test_patch()
            print("\n🎉 DLM parameters will now work with NeMo-Skills OpenAI API!")
            print("   You can now run: python xp/nemo-skills/eval_dlm.py --quick-test")
        else:
            print("\n❌ Patch failed. Please check the file manually.")
