import argparse
import sys
import time
from transformers import AutoModel, AutoTokenizer

def run_smoke_test():
    """
    Runs a quick smoke test to check for basic functionality.
    This function intentionally contains an error for the AI to fix.
    """
    print("--- Smoke Test Started ---")
    try:
        model_name = "UofT/CodeSciBERT-Chemical-V2"
        
        print(f"Attempting to load tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Attempting to load model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        
        print("Model and tokenizer loaded successfully.")
        print("--- Smoke Test PASSED ---")

    except Exception as e:
        print(f"ERROR: An exception occurred during the smoke test: {e}", file=sys.stderr)
        print("--- Smoke Test FAILED ---", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code to indicate failure

def run_full_experiment():
    """
    Runs a placeholder for the full experiment.
    This part is designed to succeed if reached.
    """
    print("--- Full Experiment Started ---")
    try:
        print("Simulating a long-running experiment...")
        # Simulate some work
        for i in range(5):
            print(f"Step {i+1}/5 completed...")
            time.sleep(1)
        
        print("Full experiment simulation finished successfully.")
        print("--- Full Experiment PASSED ---")

    except Exception as e:
        print(f"ERROR: An exception occurred during the full experiment: {e}", file=sys.stderr)
        print("--- Full Experiment FAILED ---", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with autonomous AI fixing.")
    
    # Use a mutually exclusive group to ensure only one test mode is selected
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test.")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment.")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    elif args.full_experiment:
        run_full_experiment()
