#!/usr/bin/env python
"""
Deployment Validation Script for Political Strategy Maker.

This script checks:
1. All required files exist
2. Dependencies are valid
3. Configuration is correct
4. Code has no import errors
5. AWS resources are properly configured
"""
import sys
import os
from pathlib import Path
import importlib.util

# Add backend to path
BACKEND_PATH = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_mark(passed: bool) -> str:
    return f"{GREEN}[OK]{RESET}" if passed else f"{RED}[FAIL]{RESET}"


def warn_mark() -> str:
    return f"{YELLOW}[WARN]{RESET}"


class DeploymentValidator:
    def __init__(self):
        self.root_path = Path(__file__).parent.parent
        self.backend_path = self.root_path / "backend"
        self.errors = []
        self.warnings = []
        self.passed = 0
        self.failed = 0
    
    def run_all_checks(self):
        """Run all validation checks."""
        print("=" * 70)
        print("Political Strategy Maker - Deployment Validation")
        print("=" * 70)
        
        self.check_file_structure()
        self.check_requirements()
        self.check_imports()
        self.check_configuration()
        self.check_sam_template()
        self.check_data_availability()
        
        self.print_summary()
        
        return len(self.errors) == 0
    
    def check_file_structure(self):
        """Check required files exist."""
        print("\n[FILES] Checking File Structure...")
        
        required_files = [
            "backend/app/__init__.py",
            "backend/app/main.py",
            "backend/app/config.py",
            "backend/app/models.py",
            "backend/app/services/orchestrator.py",
            "backend/app/services/llm.py",
            "backend/app/services/memory.py",
            "backend/app/services/ingest.py",
            "backend/app/services/rag/__init__.py",
            "backend/app/services/rag/political_rag.py",
            "backend/app/services/rag/knowledge_graph.py",
            "backend/app/services/agents/__init__.py",
            "backend/app/services/agents/strategic_orchestrator.py",
            "backend/app/aws/ws_chat.py",
            "backend/app/aws/ws_connect.py",
            "backend/app/aws/ws_disconnect.py",
            "backend/app/aws/ingest_handler.py",
            "backend/requirements.txt",
            "sam/template.yaml",
            "frontend/streamlit_app.py",
        ]
        
        for file_path in required_files:
            full_path = self.root_path / file_path
            exists = full_path.exists()
            status = check_mark(exists)
            print(f"  {status} {file_path}")
            
            if exists:
                self.passed += 1
            else:
                self.failed += 1
                self.errors.append(f"Missing file: {file_path}")
    
    def check_requirements(self):
        """Check requirements.txt for issues."""
        print("\n[DEPS] Checking Dependencies...")
        
        req_file = self.backend_path / "requirements.txt"
        if not req_file.exists():
            self.errors.append("requirements.txt not found")
            return
        
        content = req_file.read_text()
        
        # Check for problematic packages
        issues = []
        
        if "torch" in content:
            self.warnings.append("torch included - Lambda package will be >250MB")
            print(f"  {warn_mark()} torch included (Lambda size warning)")
        
        if "transformers" in content and "torch" not in content:
            issues.append("transformers requires torch")
        
        if "sentence-transformers" in content and "torch" not in content:
            issues.append("sentence-transformers requires torch")
        
        # Check required packages
        required = ["fastapi", "pydantic", "openai", "boto3"]
        for pkg in required:
            if pkg in content:
                print(f"  {check_mark(True)} {pkg}")
                self.passed += 1
            else:
                print(f"  {check_mark(False)} {pkg} (missing)")
                self.failed += 1
                self.errors.append(f"Missing dependency: {pkg}")
        
        # Check Lambda requirements exist
        lambda_req = self.backend_path / "requirements-lambda.txt"
        if lambda_req.exists():
            print(f"  {check_mark(True)} requirements-lambda.txt (optimized for Lambda)")
            self.passed += 1
        else:
            print(f"  {warn_mark()} requirements-lambda.txt not found")
            self.warnings.append("No Lambda-optimized requirements file")
    
    def check_imports(self):
        """Check for import errors."""
        print("\n[IMPORTS] Checking Module Imports...")
        
        modules_to_check = [
            ("app.config", "settings"),
            ("app.models", "Evidence"),
            ("app.services.llm", "get_llm"),
            ("app.services.orchestrator", "Orchestrator"),
            ("app.services.rag.political_rag", "PoliticalRAGSystem"),
            ("app.services.agents", "StrategicOrchestrator"),
        ]
        
        for module_name, attr in modules_to_check:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, attr):
                    print(f"  {check_mark(True)} {module_name}.{attr}")
                    self.passed += 1
                else:
                    print(f"  {check_mark(False)} {module_name}.{attr} (not found)")
                    self.failed += 1
                    self.errors.append(f"Missing: {module_name}.{attr}")
            except Exception as e:
                print(f"  {check_mark(False)} {module_name} (import error)")
                self.failed += 1
                self.errors.append(f"Import error in {module_name}: {str(e)[:50]}")
    
    def check_configuration(self):
        """Check configuration is valid."""
        print("\n[CONFIG] Checking Configuration...")
        
        try:
            from app.config import settings
            
            # Check API keys
            if settings.openai_api_key:
                masked = "***" + settings.openai_api_key[-4:]
                print(f"  {check_mark(True)} OPENAI_API_KEY: {masked}")
                self.passed += 1
            else:
                print(f"  {check_mark(False)} OPENAI_API_KEY not set")
                self.failed += 1
                self.errors.append("OPENAI_API_KEY not configured")
            
            # Check paths
            print(f"  {check_mark(True)} data_dir: {settings.data_dir}")
            print(f"  {check_mark(True)} index_dir: {settings.index_dir}")
            self.passed += 2
            
            # Check LLM provider
            print(f"  {check_mark(True)} llm_provider: {settings.llm_provider}")
            self.passed += 1
            
        except Exception as e:
            print(f"  {check_mark(False)} Configuration error: {e}")
            self.failed += 1
            self.errors.append(f"Configuration error: {e}")
    
    def check_sam_template(self):
        """Check SAM template is valid."""
        print("\n[AWS] Checking AWS SAM Template...")
        
        sam_file = self.root_path / "sam" / "template.yaml"
        if not sam_file.exists():
            print(f"  {check_mark(False)} template.yaml not found")
            self.errors.append("SAM template not found")
            return
        
        content = sam_file.read_text()
        
        # Check required resources
        resources = [
            ("DocumentsBucket", "S3 Documents Bucket"),
            ("SessionsTable", "DynamoDB Sessions"),
            ("MemoryTable", "DynamoDB Memory"),
            ("EntitiesTable", "DynamoDB Entities"),
            ("RestApi", "REST API Gateway"),
            ("WebSocketApi", "WebSocket API"),
            ("ChatFunction", "Chat Lambda"),
            ("IngestFunction", "Ingest Lambda"),
        ]
        
        for resource_id, desc in resources:
            if resource_id in content:
                print(f"  {check_mark(True)} {desc}")
                self.passed += 1
            else:
                print(f"  {check_mark(False)} {desc} (missing)")
                self.failed += 1
                self.errors.append(f"SAM missing resource: {resource_id}")
        
        # Check for common issues
        if "python3.11" in content:
            print(f"  {check_mark(True)} Python 3.11 runtime")
            self.passed += 1
        else:
            print(f"  {warn_mark()} Python runtime not 3.11")
            self.warnings.append("Consider using Python 3.11 runtime")
    
    def check_data_availability(self):
        """Check if political data is available."""
        print("\n[DATA] Checking Political Data...")
        
        data_paths = [
            self.root_path / "political-data",
            self.backend_path / "data",
        ]
        
        data_found = False
        for path in data_paths:
            if path.exists():
                csv_files = list(path.glob("*.csv"))
                xlsx_files = list(path.glob("*.xlsx"))
                
                if csv_files or xlsx_files:
                    print(f"  {check_mark(True)} Found data at: {path}")
                    print(f"      - CSV files: {len(csv_files)}")
                    print(f"      - XLSX files: {len(xlsx_files)}")
                    data_found = True
                    self.passed += 1
                    break
        
        if not data_found:
            print(f"  {warn_mark()} No political data found")
            self.warnings.append("Political data not ingested yet")
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"{YELLOW}Warnings: {len(self.warnings)}{RESET}")
        
        if self.errors:
            print(f"\n{RED}ERRORS:{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n{YELLOW}WARNINGS:{RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors:
            print(f"\n{GREEN}[SUCCESS] Application is ready for deployment!{RESET}")
            print("\nNext steps:")
            print("  1. Run: cd backend; python ..\\scripts\\ingest_political_data.py")
            print("  2. Test locally: .\\run_local.ps1")
            print("  3. Deploy to AWS: cd sam; sam build; sam deploy --guided")
        else:
            print(f"\n{RED}[FAILED] Please fix errors before deployment.{RESET}")


def main():
    validator = DeploymentValidator()
    success = validator.run_all_checks()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

