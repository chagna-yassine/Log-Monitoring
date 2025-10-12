"""
Setup Checker

Verifies that the project is correctly set up and ready to run.
"""

import sys
from pathlib import Path


def check_file(path: Path, description: str, required: bool = True) -> bool:
    """
    Check if a file exists.
    
    Args:
        path: Path to file
        description: Description of the file
        required: Whether the file is required
        
    Returns:
        True if exists or not required, False otherwise
    """
    exists = path.exists()
    status = "[OK]" if exists else "[!!]"
    req_text = "(required)" if required else "(optional)"
    
    if exists:
        size = path.stat().st_size
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"{status} {description}: {path} ({size_str})")
    else:
        print(f"{status} {description}: {path} {req_text}")
    
    return exists or not required


def check_directory(path: Path, description: str, create: bool = False) -> bool:
    """
    Check if a directory exists, optionally create it.
    
    Args:
        path: Path to directory
        description: Description
        create: Whether to create if missing
        
    Returns:
        True if exists or created, False otherwise
    """
    exists = path.exists()
    
    if not exists and create:
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {description}: {path} (created)")
        return True
    
    status = "[OK]" if exists else "[!!]"
    print(f"{status} {description}: {path}")
    return exists


def check_python_imports() -> bool:
    """Check if required Python packages are installed."""
    print("\nChecking Python packages...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'TQDM'),
        ('yaml', 'PyYAML'),
        ('requests', 'Requests')
    ]
    
    all_installed = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[!!] {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def main():
    """Main setup checker."""
    print("="*70)
    print("PROJECT SETUP CHECKER")
    print("="*70)
    
    all_good = True
    
    # Check core files
    print("\nCore Configuration Files:")
    all_good &= check_file(Path("config.yaml"), "Configuration", required=True)
    all_good &= check_file(Path("requirements.txt"), "Requirements", required=True)
    all_good &= check_file(Path("README.md"), "README", required=True)
    
    # Check source code
    print("\nSource Code:")
    src_files = [
        ("src/parsers/drain.py", "Drain Parser"),
        ("src/preprocessing/parser.py", "Log Parser"),
        ("src/preprocessing/template_mapper.py", "Template Mapper"),
        ("src/preprocessing/sequence_builder.py", "Sequence Builder"),
        ("src/preprocessing/data_splitter.py", "Data Splitter"),
        ("src/preprocessing/text_converter.py", "Text Converter"),
        ("src/model/inference.py", "Model Inference"),
        ("src/evaluation/metrics.py", "Metrics")
    ]
    
    for filepath, desc in src_files:
        check_file(Path(filepath), desc, required=True)
    
    # Check scripts
    print("\nScripts:")
    script_files = [
        ("scripts/download_data.py", "Data Download"),
        ("scripts/preprocess.py", "Preprocessing"),
        ("scripts/benchmark.py", "Benchmarking"),
        ("scripts/view_results.py", "View Results")
    ]
    
    for filepath, desc in script_files:
        check_file(Path(filepath), desc, required=True)
    
    # Check main runner
    print("\nMain Runner:")
    check_file(Path("run_all.py"), "Complete Pipeline Runner", required=True)
    
    # Check/create directories
    print("\nDirectories:")
    check_directory(Path("datasets"), "Datasets Directory", create=True)
    check_directory(Path("results"), "Results Directory", create=True)
    
    # Check data files (optional - will be downloaded)
    print("\nData Files (will be downloaded if missing):")
    check_file(Path("datasets/hdfs/HDFS.log"), "HDFS Log File", required=False)
    check_file(Path("datasets/hdfs/anomaly_label.csv"), "Label File", required=False)
    
    # Check processed files (optional - will be created)
    print("\nProcessed Files (will be created during preprocessing):")
    check_file(
        Path("datasets/hdfs/output/hdfs/hdfs_test.csv"),
        "Test Set",
        required=False
    )
    
    # Check Python packages
    packages_ok = check_python_imports()
    all_good &= packages_ok
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("[OK] SETUP CHECK PASSED")
        print("="*70)
        print("\nYour project is ready to run!")
        print("\nNext steps:")
        print("  1. Run: python run_all.py")
        print("     OR")
        print("  2. Run step by step:")
        print("     - python scripts/download_data.py")
        print("     - python scripts/preprocess.py")
        print("     - python scripts/benchmark.py")
    else:
        print("[!!] SETUP CHECK FAILED")
        print("="*70)
        print("\nPlease fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Check file paths are correct")
        print("  - Ensure you're in the project root directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

