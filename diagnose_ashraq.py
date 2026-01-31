#!/usr/bin/env python3

"""
Diagnose ashraq/financial-news Dataset Structure
================================================

This script helps identify the actual field names and structure
of the ashraq dataset so we can properly extract data.

Run this first to understand the dataset, then update the main script.
"""

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import json

print("=" * 80)
print("DIAGNOSING ASHRAQ DATASET STRUCTURE")
print("=" * 80)

try:
    print("\nLoading ashraq/financial-news dataset...")
    ds = load_dataset("ashraq/financial-news", split="train", streaming=True)
    
    print("Scanning first 100 rows to identify field names and values...\n")
    
    sample_rows = []
    for i, row in enumerate(tqdm(ds, total=100, desc="ashraq")):
        if i >= 100:
            break
        sample_rows.append(row)
    
    if sample_rows:
        print("\n" + "=" * 80)
        print("DATASET STRUCTURE")
        print("=" * 80)
        
        first_row = sample_rows[0]
        print(f"\nTotal fields: {len(first_row)}")
        print(f"\nAvailable fields:")
        for key in sorted(first_row.keys()):
            value = first_row[key]
            value_type = type(value).__name__
            value_preview = str(value)[:100] if value else "None/Empty"
            print(f"  • {key:25s} ({value_type:10s})")
        
        # Detailed inspection
        print("\n" + "=" * 80)
        print("SAMPLE DATA (First 5 Rows)")
        print("=" * 80)
        
        for i, row in enumerate(sample_rows[:5]):
            print(f"\n{'─' * 80}")
            print(f"ROW {i}:")
            print(f"{'─' * 80}")
            
            for key in sorted(row.keys()):
                value = row[key]
                if isinstance(value, str):
                    preview = value[:150]
                    if len(value) > 150:
                        preview += "..."
                    print(f"  {key:25s}: {preview}")
                else:
                    print(f"  {key:25s}: {value}")
        
        # Check for date-related fields
        print("\n" + "=" * 80)
        print("DATE FIELD ANALYSIS")
        print("=" * 80)
        
        date_fields = {
            'date': [],
            'published': [],
            'publish_date': [],
            'published_at': [],
            'created': [],
            'timestamp': [],
            'time': []
        }
        
        for field_name in first_row.keys():
            if any(date_keyword in field_name.lower() for date_keyword in ['date', 'time', 'published', 'publish']):
                print(f"\n✓ Found potential date field: {field_name}")
                # Collect sample values
                for row in sample_rows[:10]:
                    if row.get(field_name):
                        print(f"  Sample: {row[field_name]}")
                        break
        
        # Check for text-related fields
        print("\n" + "=" * 80)
        print("TEXT FIELD ANALYSIS")
        print("=" * 80)
        
        text_fields = []
        for field_name in first_row.keys():
            if any(text_keyword in field_name.lower() for text_keyword in ['text', 'content', 'body', 'article', 'headline', 'title', 'description', 'summary']):
                text_fields.append(field_name)
                print(f"\n✓ Found potential text field: {field_name}")
                # Get field length info
                lengths = []
                for row in sample_rows:
                    if row.get(field_name):
                        lengths.append(len(str(row[field_name])))
                
                if lengths:
                    print(f"  Average length: {sum(lengths) // len(lengths)} chars")
                    print(f"  Max length: {max(lengths)} chars")
                    print(f"  Sample: {str(sample_rows[0].get(field_name))[:100]}...")
        
        # Summary recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR fetch_recent_news_2024_2025.py")
        print("=" * 80)
        
        # Find date field
        date_field = None
        for field in first_row.keys():
            if any(x in field.lower() for x in ['date', 'published', 'publish', 'time']):
                date_field = field
                break
        
        # Find text fields
        text_field = None
        for field in first_row.keys():
            if field.lower() in ['content', 'body', 'article', 'text']:
                text_field = field
                break
        
        if not text_field:
            for field in first_row.keys():
                if any(x in field.lower() for x in ['title', 'headline', 'summary']):
                    text_field = field
                    break
        
        print(f"\n1. Use '{date_field}' as the DATE field")
        print(f"2. Use '{text_field}' as the TEXT field")
        
        # Save findings to JSON
        findings = {
            'all_fields': list(first_row.keys()),
            'date_field': date_field,
            'text_field': text_field,
            'date_field_sample': str(sample_rows[0].get(date_field)),
            'text_field_sample': str(sample_rows[0].get(text_field))[:200]
        }
        
        with open('data/news_articles/ashraq_structure.json', 'w') as f:
            json.dump(findings, f, indent=2)
        
        print(f"\n✓ Findings saved to: data/news_articles/ashraq_structure.json")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Use the findings above to update fetch_recent_news_2024_2025.py")
print("=" * 80)
