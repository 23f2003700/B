#!/usr/bin/env python3
"""
Data Structure Explorer for Redtape Business Analysis
Run this first to understand your data structure
"""

import pandas as pd
import os

def explore_excel_structure():
    """Explore the structure of all Excel files"""
    
    files = [
        'DEC 2024.xlsx',
        'Jan 2025.xlsx', 
        'Feb 2025.xlsx',
        'March 2025.xlsx',
        'April 2025.xlsx',
        'May 2025.xlsx'
    ]
    
    print("EXCEL FILES STRUCTURE ANALYSIS")
    print("="*50)
    
    for file in files:
        if os.path.exists(file):
            print(f"\n📊 ANALYZING: {file}")
            print("-" * 30)
            
            try:
                # Get all sheet names
                xl_file = pd.ExcelFile(file)
                sheet_names = xl_file.sheet_names
                print(f"Sheets: {sheet_names}")
                
                # Analyze each sheet
                for sheet in sheet_names:
                    print(f"\n  📋 Sheet: {sheet}")
                    df = pd.read_excel(file, sheet_name=sheet)
                    
                    print(f"    Shape: {df.shape}")
                    print(f"    Columns: {list(df.columns)}")
                    
                    # Show sample data
                    print(f"    Sample data:")
                    for i, row in df.head(3).iterrows():
                        print(f"      Row {i}: {dict(row)}")
                    
                    # Data types
                    print(f"    Data types: {dict(df.dtypes)}")
                    
                    # Non-null counts
                    print(f"    Non-null counts: {dict(df.count())}")
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"❌ File not found: {file}")

if __name__ == "__main__":
    explore_excel_structure()
