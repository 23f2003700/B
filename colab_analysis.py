#!/usr/bin/env python3
"""
Google Colab Execution Script for Redtape Business Analysis
This script is optimized to run on Google Colab environment
"""

# Install required packages in Colab
print("📦 Installing required packages...")
print("="*50)

import subprocess
import sys

# Install packages
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'openpyxl', 'xlrd', 'scipy']
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

print("✅ All packages installed successfully!")

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n🚀 REDTAPE STORE JODHPUR - BUSINESS ANALYSIS")
print("="*60)
print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

class RedtapeColabAnalysis:
    def __init__(self):
        self.files = [
            'DEC 2024.xlsx',
            'Jan 2025.xlsx', 
            'Feb 2025.xlsx',
            'March 2025.xlsx',
            'April 2025.xlsx',
            'May 2025.xlsx'
        ]
        self.data = {}
        self.combined_df = None
        self.results = {}
    
    def load_files(self):
        """Load all Excel files with error handling"""
        print("\n📊 Loading Excel files...")
        
        successful_loads = 0
        for file in self.files:
            try:
                # Try to read the Excel file
                df = pd.read_excel(file)
                
                # Extract month from filename
                month = file.replace('.xlsx', '').strip()
                df['Month_Year'] = month
                
                # Store the data
                self.data[month] = df
                
                print(f"✅ {file}: {len(df)} rows, {len(df.columns)} columns")
                print(f"   Columns: {list(df.columns)}")
                
                # Show sample data
                print(f"   Sample data preview:")
                for i, (idx, row) in enumerate(df.head(2).iterrows()):
                    print(f"     Row {i+1}: {dict(row)}")
                
                successful_loads += 1
                
            except FileNotFoundError:
                print(f"❌ File not found: {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {str(e)}")
        
        print(f"\n📈 Successfully loaded {successful_loads}/{len(self.files)} files")
        
        # Combine all data
        if self.data:
            self.combined_df = pd.concat(self.data.values(), ignore_index=True)
            print(f"📊 Combined dataset: {len(self.combined_df)} total rows")
        
        return successful_loads > 0
    
    def analyze_data_structure(self):
        """Detailed data structure analysis"""
        if self.combined_df is None:
            print("❌ No data to analyze")
            return
        
        print("\n" + "="*60)
        print("📊 DETAILED DATA STRUCTURE ANALYSIS")
        print("="*60)
        
        # Basic information
        print(f"\n📋 Dataset Overview:")
        print(f"   • Total Records: {len(self.combined_df):,}")
        print(f"   • Total Columns: {len(self.combined_df.columns)}")
        print(f"   • Months Covered: {sorted(self.combined_df['Month_Year'].unique())}")
        
        # Column analysis
        print(f"\n📊 Column Details:")
        for i, col in enumerate(self.combined_df.columns, 1):
            dtype = str(self.combined_df[col].dtype)
            non_null = self.combined_df[col].count()
            null_count = len(self.combined_df) - non_null
            unique_vals = self.combined_df[col].nunique()
            
            print(f"   {i:2d}. {col}")
            print(f"       Type: {dtype}, Non-null: {non_null:,}, Missing: {null_count}, Unique: {unique_vals:,}")
            
            # Show sample values
            if dtype == 'object':
                sample_vals = self.combined_df[col].dropna().unique()[:5]
                print(f"       Sample values: {list(sample_vals)}")
            else:
                stats = self.combined_df[col].describe()
                print(f"       Range: [{stats['min']:.2f}, {stats['max']:.2f}], Mean: {stats['mean']:.2f}")
        
        # Identify potential business metrics
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Month_Year' in numeric_cols:
            numeric_cols.remove('Month_Year')
        
        print(f"\n💰 Numeric Columns (Potential Business Metrics): {len(numeric_cols)}")
        for col in numeric_cols:
            print(f"   • {col}")
        
        # Store results
        self.results['data_structure'] = {
            'total_records': len(self.combined_df),
            'total_columns': len(self.combined_df.columns),
            'months': sorted(self.combined_df['Month_Year'].unique()),
            'numeric_columns': numeric_cols,
            'column_info': {}
        }
        
        for col in self.combined_df.columns:
            self.results['data_structure']['column_info'][col] = {
                'dtype': str(self.combined_df[col].dtype),
                'non_null_count': int(self.combined_df[col].count()),
                'unique_values': int(self.combined_df[col].nunique())
            }
    
    def perform_business_analysis(self):
        """Comprehensive business analysis"""
        if self.combined_df is None:
            print("❌ No data for analysis")
            return
        
        print("\n" + "="*60)
        print("💼 BUSINESS PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Get numeric columns for analysis
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Month_Year' in numeric_cols:
            numeric_cols.remove('Month_Year')
        
        if not numeric_cols:
            print("⚠️ No numeric columns found for business analysis")
            return
        
        # Monthly aggregation
        print(f"\n📈 Monthly Business Metrics Summary:")
        monthly_summary = self.combined_df.groupby('Month_Year')[numeric_cols].agg(['sum', 'mean', 'count']).round(2)
        print(monthly_summary)
        
        # Calculate growth rates
        print(f"\n📊 Month-over-Month Growth Analysis:")
        for col in numeric_cols[:5]:  # Top 5 columns
            monthly_totals = self.combined_df.groupby('Month_Year')[col].sum()
            growth_rates = monthly_totals.pct_change() * 100
            
            print(f"\n   📈 {col} Growth Rates:")
            for month, growth in growth_rates.items():
                if pd.notna(growth):
                    trend = "📈" if growth > 0 else "📉" if growth < 0 else "➡️"
                    print(f"      {month}: {growth:+.2f}% {trend}")
        
        # Overall performance metrics
        print(f"\n🎯 Overall Performance Metrics:")
        
        for col in numeric_cols[:3]:  # Top 3 columns
            total = self.combined_df[col].sum()
            average = self.combined_df[col].mean()
            std_dev = self.combined_df[col].std()
            
            print(f"\n   💰 {col}:")
            print(f"      Total: {total:,.2f}")
            print(f"      Average: {average:,.2f}")
            print(f"      Std Dev: {std_dev:,.2f}")
            print(f"      Coefficient of Variation: {(std_dev/average)*100:.2f}%")
        
        # Store results
        self.results['business_analysis'] = {
            'monthly_summary': monthly_summary.to_dict(),
            'numeric_columns_analyzed': numeric_cols,
            'total_metrics': {}
        }
        
        for col in numeric_cols:
            self.results['business_analysis']['total_metrics'][col] = {
                'total': float(self.combined_df[col].sum()),
                'average': float(self.combined_df[col].mean()),
                'std_dev': float(self.combined_df[col].std())
            }
    
    def create_visualizations(self):
        """Create business visualizations"""
        if self.combined_df is None:
            print("❌ No data for visualizations")
            return
        
        print("\n" + "="*60)
        print("📊 CREATING BUSINESS VISUALIZATIONS")
        print("="*60)
        
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Month_Year' in numeric_cols:
            numeric_cols.remove('Month_Year')
        
        if not numeric_cols:
            print("⚠️ No numeric columns for visualization")
            return
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Redtape Store Jodhpur - Business Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # Monthly trends for top columns
        for i, col in enumerate(numeric_cols[:4]):
            row, col_idx = divmod(i, 3)
            if row < 2:
                monthly_data = self.combined_df.groupby('Month_Year')[col].sum()
                
                axes[row, col_idx].bar(range(len(monthly_data)), monthly_data.values, 
                                      color=plt.cm.Set3(np.linspace(0, 1, len(monthly_data))))
                axes[row, col_idx].set_title(f'Monthly {col}', fontweight='bold')
                axes[row, col_idx].set_xlabel('Month')
                axes[row, col_idx].set_ylabel(col)
                axes[row, col_idx].set_xticks(range(len(monthly_data)))
                axes[row, col_idx].set_xticklabels(monthly_data.index, rotation=45)
        
        # Distribution plot
        if len(numeric_cols) > 0:
            row, col_idx = 1, 2
            axes[row, col_idx].hist(self.combined_df[numeric_cols[0]], bins=20, alpha=0.7, color='skyblue')
            axes[row, col_idx].set_title(f'Distribution of {numeric_cols[0]}', fontweight='bold')
            axes[row, col_idx].set_xlabel(numeric_cols[0])
            axes[row, col_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('redtape_business_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Dashboard saved as 'redtape_business_dashboard.png'")
        
        # Create trend analysis
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            monthly_comparison = self.combined_df.groupby('Month_Year')[numeric_cols[:3]].sum()
            monthly_comparison.plot(kind='line', marker='o', ax=ax, linewidth=2)
            
            ax.set_title('Monthly Business Trends - Key Metrics', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Value')
            ax.legend(numeric_cols[:3])
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('monthly_trends.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Trend analysis saved as 'monthly_trends.png'")
    
    def generate_insights(self):
        """Generate business insights and recommendations"""
        print("\n" + "="*60)
        print("🎯 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        if self.combined_df is None:
            print("❌ No data for insights generation")
            return
        
        # Calculate key insights
        insights = {
            'data_quality': {},
            'performance_insights': {},
            'recommendations': []
        }
        
        # Data quality assessment
        total_cells = len(self.combined_df) * len(self.combined_df.columns)
        non_null_cells = self.combined_df.count().sum()
        data_completeness = (non_null_cells / total_cells) * 100
        
        insights['data_quality'] = {
            'completeness_score': data_completeness,
            'total_records': len(self.combined_df),
            'months_covered': len(self.combined_df['Month_Year'].unique())
        }
        
        print(f"📊 Data Quality Assessment:")
        print(f"   • Data Completeness: {data_completeness:.1f}%")
        print(f"   • Records Quality: {'Excellent' if data_completeness > 90 else 'Good' if data_completeness > 75 else 'Needs Improvement'}")
        
        # Performance insights
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Month_Year' in numeric_cols:
            numeric_cols.remove('Month_Year')
        
        if numeric_cols:
            # Calculate trends
            monthly_data = self.combined_df.groupby('Month_Year')[numeric_cols[0]].sum()
            if len(monthly_data) > 1:
                overall_trend = (monthly_data.iloc[-1] - monthly_data.iloc[0]) / monthly_data.iloc[0] * 100
                insights['performance_insights']['overall_growth'] = overall_trend
                
                print(f"\n📈 Performance Insights:")
                print(f"   • Overall Growth Trend: {overall_trend:+.2f}%")
                print(f"   • Performance Rating: {'Strong Growth' if overall_trend > 10 else 'Moderate Growth' if overall_trend > 0 else 'Declining'}")
        
        # Generate recommendations
        recommendations = [
            "🎯 Implement monthly performance monitoring dashboard",
            "📊 Establish data quality standards and validation processes", 
            "💡 Develop predictive analytics for demand forecasting",
            "🔍 Focus on high-performing periods for strategy replication",
            "📈 Consider seasonal adjustment strategies for inventory management",
            "👥 Implement customer segmentation for targeted marketing",
            "💰 Optimize pricing strategies based on performance patterns",
            "🚀 Invest in data automation tools for real-time insights"
        ]
        
        insights['recommendations'] = recommendations
        
        print(f"\n🎯 Strategic Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Store insights
        self.results['insights'] = insights
        
        return insights
    
    def export_results(self):
        """Export all analysis results"""
        print(f"\n💾 Exporting Analysis Results...")
        
        # Save detailed results
        with open('redtape_analysis_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary report
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(self.combined_df) if self.combined_df is not None else 0,
                'months_analyzed': len(self.combined_df['Month_Year'].unique()) if self.combined_df is not None else 0,
                'data_quality_score': self.results.get('insights', {}).get('data_quality', {}).get('completeness_score', 0)
            },
            'key_findings': self.results.get('insights', {}).get('recommendations', [])[:5],
            'files_processed': len([k for k, v in self.data.items() if v is not None])
        }
        
        with open('analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✅ Results exported:")
        print("   • redtape_analysis_results.json - Detailed analysis results")
        print("   • analysis_summary.json - Executive summary")
        
        return summary

# Main execution
def run_colab_analysis():
    """Main function to run complete analysis in Colab"""
    
    analyzer = RedtapeColabAnalysis()
    
    # Step 1: Load data
    if analyzer.load_files():
        print("\n✅ Data loading successful - proceeding with analysis...")
        
        # Step 2: Analyze structure
        analyzer.analyze_data_structure()
        
        # Step 3: Business analysis
        analyzer.perform_business_analysis()
        
        # Step 4: Create visualizations
        analyzer.create_visualizations()
        
        # Step 5: Generate insights
        insights = analyzer.generate_insights()
        
        # Step 6: Export results
        summary = analyzer.export_results()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📊 Generated outputs:")
        print("   • redtape_business_dashboard.png")
        print("   • monthly_trends.png") 
        print("   • redtape_analysis_results.json")
        print("   • analysis_summary.json")
        
        print(f"\n🎯 Ready for Final Report Integration!")
        print(f"   • Data Quality Score: {insights['data_quality']['completeness_score']:.1f}%")
        print(f"   • Records Analyzed: {insights['data_quality']['total_records']:,}")
        print(f"   • Time Period: {insights['data_quality']['months_covered']} months")
        
        return analyzer, insights, summary
    
    else:
        print("\n❌ Analysis cannot proceed - please check your Excel files")
        print("📋 Instructions:")
        print("   1. Upload your Excel files to Colab environment")
        print("   2. Ensure files are named exactly as expected:")
        for file in analyzer.files:
            print(f"      • {file}")
        print("   3. Re-run this script after uploading files")
        
        return None, None, None

# Execute the analysis
if __name__ == "__main__":
    analyzer, insights, summary = run_colab_analysis()
