"""
Batch Training Script for All Image-Based Cancer Models
Trains both Blood Cell Cancer and Skin Cancer models sequentially
"""

import subprocess
import sys
import time
from datetime import datetime

def run_training_script(script_name, description):
    """Run a training script and track time"""
    print("\n" + "="*80)
    print(f"Starting: {description}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "="*80)
        print(f"✓ Completed: {description}")
        print(f"Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print("="*80 + "\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print(f"✗ Failed: {description}")
        print(f"Error: {e}")
        print("="*80 + "\n")
        return False

def main():
    """Main batch training pipeline"""
    print("="*80)
    print("MULTI-CANCER DETECTION SYSTEM - BATCH TRAINING")
    print("Training all image-based cancer detection models")
    print("="*80)
    
    overall_start = time.time()
    
    # Training schedule
    training_tasks = [
        {
            'script': 'train_blood_cancer_model.py',
            'description': 'Blood Cell Cancer Detection Model (4-class)',
            'priority': 1
        },
        {
            'script': 'train_skin_cancer_model.py',
            'description': 'Skin Cancer Detection Model (7-class HAM10000)',
            'priority': 2
        }
    ]
    
    results = {}
    
    # Execute training tasks
    for task in training_tasks:
        success = run_training_script(task['script'], task['description'])
        results[task['description']] = 'SUCCESS' if success else 'FAILED'
        
        if not success:
            print(f"\n⚠ Warning: {task['description']} failed!")
            user_input = input("Continue with remaining tasks? (y/n): ")
            if user_input.lower() != 'y':
                print("Training pipeline aborted by user.")
                break
    
    # Summary
    overall_elapsed = time.time() - overall_start
    hours = int(overall_elapsed // 3600)
    minutes = int((overall_elapsed % 3600) // 60)
    seconds = int(overall_elapsed % 60)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE SUMMARY")
    print("="*80)
    
    for description, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {description}: {status}")
    
    print(f"\nTotal time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check if all succeeded
    all_success = all(status == 'SUCCESS' for status in results.values())
    
    if all_success:
        print("\n🎉 All models trained successfully!")
        print("\nNext steps:")
        print("1. Review model metrics in *_results.json files")
        print("2. Check training plots: *_training_history.png")
        print("3. Examine confusion matrices: *_confusion_matrix.png")
        print("4. Proceed with multi-cancer pipeline integration")
    else:
        print("\n⚠ Some models failed to train. Please review errors above.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
