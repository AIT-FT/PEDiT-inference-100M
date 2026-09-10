import os
import sys
import subprocess
import importlib.util

# Mapping from pip requirement name to import name
PKG_IMPORT_MAP = {
    "pillow": "PIL",
    "python-dotenv": "dotenv",
    "python-multipart": "multipart",
    "protobuf": "google.protobuf",
    "huggingface-hub": "huggingface_hub",
    "huggingface_hub": "huggingface_hub",
}

def is_module_installed(module_name: str) -> bool:
    try:
        if "." in module_name:
            top_level = module_name.split(".")[0]
            if importlib.util.find_spec(top_level) is None:
                return False
            # Check nested module import
            importlib.import_module(module_name)
            return True
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False

def ensure_dependencies(req_file_name: str = "requirements_inference.txt") -> bool:
    """
    Checks if all dependencies listed in req_file_name are installed.
    If any dependency is missing, automatically installs it via pip.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    req_path = os.path.join(root_dir, req_file_name)
    
    if not os.path.exists(req_path):
        print(f"[check_deps] Warning: {req_path} not found. Skipping auto-check.")
        return True

    missing_packages = []
    
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Extract package name (e.g. "Pillow>=9.5.0" -> "Pillow")
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].split("~=")[0].strip()
            import_name = PKG_IMPORT_MAP.get(pkg_name.lower(), pkg_name.replace("-", "_"))
            
            if not is_module_installed(import_name):
                missing_packages.append(line)

    if missing_packages:
        print(f"[check_deps] Missing dependencies detected ({len(missing_packages)}): {', '.join(missing_packages)}")
        print(f"[check_deps] Installing missing dependencies from {req_file_name}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
            subprocess.check_call(cmd)
            print("[check_deps] All dependencies successfully installed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[check_deps] Error during package installation: {e}")
            sys.exit(1)
    
    return True

if __name__ == "__main__":
    ensure_dependencies()
