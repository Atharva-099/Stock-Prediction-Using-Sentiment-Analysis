#!/usr/bin/env python3
"""
Quick Data Availability Check Script
=====================================
Run this first to see what years of data are available before training.

Usage:
    python check_data_availability.py

"""

import sys
sys.path.insert(0, '.')

import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HuggingFace token
HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')


def main():
    print("\n" + "=" * 80)
    print(" 📊 DATA AVAILABILITY CHECK: Financial News (1999-2025)")
    print("=" * 80)
    
    from src.data_availability_checker import DataAvailabilityChecker
    
    checker = DataAvailabilityChecker(hf_token=HF_TOKEN)
    
    # Run full check
    report = checker.generate_full_report(sample_size=20000)
    
    # Save report
    os.makedirs('results', exist_ok=True)
    checker.save_report('results/data_availability_report.json')
    
    # Print summary
    print("\n" + "=" * 80)
    print(" 📋 QUICK SUMMARY")
    print("=" * 80)
    
    if 'summary' in report:
        summary = report['summary']
        
        print(f"\n📅 Years with data available:")
        for year in sorted(summary.get('all_years', [])):
            sources = summary.get('year_sources', {}).get(year, [])
            print(f"   {year}: {', '.join(sources)}")
        
        print(f"\n🎯 Recommended Training Strategy:")
        print(f"   Training years: {summary.get('train_years', 'N/A')}")
        print(f"   Fine-tuning years: {summary.get('finetune_years', 'N/A')}")
        print(f"   Validation year: {summary.get('validation_year', 'N/A')}")
    
    print("\n✅ Full report saved to: results/data_availability_report.json")
    print("\n💡 Next step: Run 'python train_historical_model.py' to start training!")
    
    return report


if __name__ == "__main__":
    main()


