#!/usr/bin/env python3
"""
Comprehensive Retail Business Analysis for Redtape Store Jodhpur
Advanced Analytics and Visualization Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class RedtapeRetailAnalytics:
    def __init__(self):
        self.data_files = [
            'DEC 2024.xlsx',
            'Jan 2025.xlsx', 
            'Feb 2025.xlsx',
            'March 2025.xlsx',
            'April 2025.xlsx',
            'May 2025.xlsx'
        ]
        self.monthly_data = {}
        self.combined_df = None
        self.analysis_results = {}
        
    def load_and_prepare_data(self):
        """Load all Excel files and prepare for analysis"""
        print("📊 Loading and preparing data...")
        
        for file in self.data_files:
            try:
                # Load Excel file (try multiple sheet approaches)
                xl = pd.ExcelFile(file)
                
                # Get first sheet if multiple sheets exist
                sheet_name = xl.sheet_names[0]
                df = pd.read_excel(file, sheet_name=sheet_name)
                
                # Extract month from filename
                month_year = file.replace('.xlsx', '').strip()
                
                # Add temporal columns
                df['Month_Year'] = month_year
                df['File_Source'] = file
                
                # Convert date columns if they exist
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                for col in date_columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
                
                self.monthly_data[month_year] = df
                print(f"✅ Loaded {file}: {df.shape[0]} records, {df.shape[1]} columns")
                
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        # Combine all data
        if self.monthly_data:
            self.combined_df = pd.concat(self.monthly_data.values(), ignore_index=True)
            print(f"\n📈 Combined dataset: {self.combined_df.shape[0]} total records")
            
        return self.combined_df
    
    def data_profiling(self):
        """Comprehensive data profiling and quality assessment"""
        if self.combined_df is None:
            print("No data loaded!")
            return
            
        print("\n" + "="*60)
        print("📊 DATA PROFILING REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"📋 Dataset Overview:")
        print(f"   • Total Records: {len(self.combined_df):,}")
        print(f"   • Total Columns: {len(self.combined_df.columns)}")
        print(f"   • Memory Usage: {self.combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"   • Date Range: {self.combined_df['Month_Year'].unique()}")
        
        # Column analysis
        print(f"\n📊 Column Analysis:")
        for col in self.combined_df.columns:
            non_null = self.combined_df[col].count()
            null_pct = (len(self.combined_df) - non_null) / len(self.combined_df) * 100
            dtype = self.combined_df[col].dtype
            unique_vals = self.combined_df[col].nunique()
            
            print(f"   • {col}: {dtype}, {unique_vals:,} unique, {null_pct:.1f}% missing")
        
        # Identify potential business metrics columns
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\n💰 Potential Business Metrics ({len(numeric_cols)} columns):")
        for col in numeric_cols:
            if col not in ['Month_Year']:
                stats = self.combined_df[col].describe()
                print(f"   • {col}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        
        return {
            'total_records': len(self.combined_df),
            'columns': list(self.combined_df.columns),
            'numeric_columns': numeric_cols,
            'date_range': self.combined_df['Month_Year'].unique().tolist()
        }
    
    def sales_performance_analysis(self):
        """Comprehensive sales analysis"""
        print("\n" + "="*60)
        print("💰 SALES PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Identify sales-related columns
        sales_cols = [col for col in self.combined_df.columns 
                     if any(keyword in col.lower() for keyword in 
                           ['sales', 'revenue', 'amount', 'total', 'price', 'value'])]
        
        print(f"🔍 Identified sales columns: {sales_cols}")
        
        if not sales_cols:
            print("⚠️ No clear sales columns identified. Using all numeric columns.")
            sales_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Monthly aggregation
        monthly_summary = self.combined_df.groupby('Month_Year').agg({
            col: ['sum', 'mean', 'count', 'std'] for col in sales_cols[:5]  # Limit to top 5 columns
        }).round(2)
        
        print(f"\n📈 Monthly Sales Summary:")
        print(monthly_summary)
        
        # Create comprehensive sales visualizations
        self.create_sales_dashboard(sales_cols)
        
        return monthly_summary
    
    def customer_analytics(self):
        """Customer behavior and segmentation analysis"""
        print("\n" + "="*60)
        print("👥 CUSTOMER ANALYTICS")
        print("="*60)
        
        # Identify customer-related columns
        customer_cols = [col for col in self.combined_df.columns 
                        if any(keyword in col.lower() for keyword in 
                              ['customer', 'client', 'buyer', 'user', 'member'])]
        
        if customer_cols:
            print(f"🔍 Customer columns found: {customer_cols}")
            
            # Customer analysis
            for col in customer_cols[:3]:  # Analyze top 3 customer columns
                if self.combined_df[col].dtype == 'object':
                    unique_customers = self.combined_df[col].nunique()
                    print(f"   • {col}: {unique_customers:,} unique customers")
                    
                    # Monthly customer trends
                    monthly_customers = self.combined_df.groupby('Month_Year')[col].nunique()
                    print(f"   • Monthly trend: {monthly_customers.to_dict()}")
        else:
            print("⚠️ No clear customer columns identified")
            # Use transaction count as customer proxy
            monthly_transactions = self.combined_df.groupby('Month_Year').size()
            print(f"📊 Monthly transaction counts: {monthly_transactions.to_dict()}")
    
    def inventory_analysis(self):
        """Inventory and product performance analysis"""
        print("\n" + "="*60)
        print("📦 INVENTORY & PRODUCT ANALYSIS")
        print("="*60)
        
        # Identify product/inventory columns
        product_cols = [col for col in self.combined_df.columns 
                       if any(keyword in col.lower() for keyword in 
                             ['product', 'item', 'sku', 'stock', 'inventory', 'quantity'])]
        
        print(f"🔍 Product/Inventory columns: {product_cols}")
        
        if product_cols:
            for col in product_cols[:3]:
                if self.combined_df[col].dtype == 'object':
                    # Product variety analysis
                    unique_products = self.combined_df[col].nunique()
                    top_products = self.combined_df[col].value_counts().head(10)
                    
                    print(f"\n📊 {col} Analysis:")
                    print(f"   • Total unique products: {unique_products:,}")
                    print(f"   • Top 10 products:")
                    for product, count in top_products.items():
                        print(f"     - {product}: {count} records")
    
    def financial_analysis(self):
        """Financial performance and profitability analysis"""
        print("\n" + "="*60)
        print("💰 FINANCIAL ANALYSIS")
        print("="*60)
        
        # Identify financial columns
        financial_cols = [col for col in self.combined_df.columns 
                         if any(keyword in col.lower() for keyword in 
                               ['profit', 'cost', 'margin', 'expense', 'revenue', 'income'])]
        
        print(f"🔍 Financial columns: {financial_cols}")
        
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Monthly financial trends
            monthly_financials = self.combined_df.groupby('Month_Year')[numeric_cols].sum()
            
            print(f"\n📈 Monthly Financial Summary:")
            print(monthly_financials.round(2))
            
            # Calculate growth rates
            print(f"\n📊 Month-over-Month Growth Rates:")
            for col in numeric_cols[:5]:
                pct_change = monthly_financials[col].pct_change() * 100
                print(f"   • {col}: {pct_change.round(2).to_dict()}")
    
    def create_sales_dashboard(self, sales_cols):
        """Create comprehensive sales visualization dashboard"""
        
        # Setup the subplot grid
        fig = plt.figure(figsize=(20, 15))
        
        # Monthly trends for top sales columns
        for i, col in enumerate(sales_cols[:4]):
            plt.subplot(2, 3, i+1)
            
            monthly_data = self.combined_df.groupby('Month_Year')[col].sum()
            
            # Create bar plot
            bars = plt.bar(range(len(monthly_data)), monthly_data.values, 
                          color=plt.cm.Set3(np.linspace(0, 1, len(monthly_data))))
            
            plt.title(f'Monthly {col}', fontsize=12, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel(col)
            plt.xticks(range(len(monthly_data)), monthly_data.index, rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, monthly_data.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(monthly_data.values)*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Distribution analysis
        plt.subplot(2, 3, 5)
        if len(sales_cols) > 0:
            self.combined_df[sales_cols[0]].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {sales_cols[0]}', fontsize=12, fontweight='bold')
            plt.xlabel(sales_cols[0])
            plt.ylabel('Frequency')
        
        # Monthly comparison
        plt.subplot(2, 3, 6)
        if len(sales_cols) >= 2:
            monthly_comparison = self.combined_df.groupby('Month_Year')[sales_cols[:2]].sum()
            monthly_comparison.plot(kind='bar', ax=plt.gca())
            plt.title('Monthly Comparison - Top 2 Metrics', fontsize=12, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel('Value')
            plt.legend(sales_cols[:2], loc='upper left')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Sales dashboard saved as 'sales_dashboard.png'")
    
    def create_business_insights_report(self):
        """Generate comprehensive business insights"""
        print("\n" + "="*60)
        print("🎯 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        insights = {
            'data_quality': {},
            'sales_performance': {},
            'trends': {},
            'recommendations': []
        }
        
        # Data quality insights
        insights['data_quality'] = {
            'total_records': len(self.combined_df),
            'date_coverage': len(self.combined_df['Month_Year'].unique()),
            'data_completeness': (self.combined_df.count().sum() / (len(self.combined_df) * len(self.combined_df.columns))) * 100
        }
        
        print(f"📊 Data Quality Score: {insights['data_quality']['data_completeness']:.1f}%")
        
        # Sales trends
        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            monthly_totals = self.combined_df.groupby('Month_Year')[numeric_cols[0]].sum()
            
            # Calculate overall trend
            if len(monthly_totals) > 1:
                trend = (monthly_totals.iloc[-1] - monthly_totals.iloc[0]) / monthly_totals.iloc[0] * 100
                insights['trends']['overall_growth'] = trend
                
                print(f"📈 Overall Growth Trend: {trend:.2f}%")
        
        # Generate recommendations
        recommendations = [
            "🎯 Implement data standardization across all monthly files",
            "📊 Establish regular monthly performance review meetings",
            "💡 Consider implementing automated data collection systems",
            "🔍 Focus on high-performing months for strategy replication",
            "📈 Develop predictive models for future sales forecasting"
        ]
        
        insights['recommendations'] = recommendations
        
        print(f"\n🎯 Key Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
        
        return insights
    
    def export_analysis_results(self):
        """Export analysis results for report inclusion"""
        
        # Create summary statistics
        summary_stats = {}
        
        if self.combined_df is not None:
            # Basic statistics
            summary_stats['basic_stats'] = {
                'total_records': len(self.combined_df),
                'total_columns': len(self.combined_df.columns),
                'months_covered': self.combined_df['Month_Year'].nunique(),
                'date_range': list(self.combined_df['Month_Year'].unique())
            }
            
            # Numeric summary
            numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary_stats['numeric_summary'] = self.combined_df[numeric_cols].describe().to_dict()
            
            # Monthly aggregations
            summary_stats['monthly_aggregations'] = {}
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                monthly_data = self.combined_df.groupby('Month_Year')[col].agg(['sum', 'mean', 'count']).to_dict()
                summary_stats['monthly_aggregations'][col] = monthly_data
        
        # Save to file
        import json
        with open('analysis_results.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"\n💾 Analysis results exported to 'analysis_results.json'")
        
        return summary_stats

# Main execution function
def run_comprehensive_analysis():
    """Run the complete business analysis pipeline"""
    
    print("🚀 REDTAPE STORE JODHPUR - COMPREHENSIVE BUSINESS ANALYSIS")
    print("="*70)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    # Initialize analyzer
    analyzer = RedtapeRetailAnalytics()
    
    # Step 1: Load and prepare data
    data = analyzer.load_and_prepare_data()
    
    if data is not None and len(data) > 0:
        print(f"\n✅ Data loaded successfully: {len(data)} records")
        
        # Step 2: Data profiling
        profile_results = analyzer.data_profiling()
        
        # Step 3: Sales analysis
        sales_results = analyzer.sales_performance_analysis()
        
        # Step 4: Customer analytics
        analyzer.customer_analytics()
        
        # Step 5: Inventory analysis
        analyzer.inventory_analysis()
        
        # Step 6: Financial analysis
        analyzer.financial_analysis()
        
        # Step 7: Generate insights
        insights = analyzer.create_business_insights_report()
        
        # Step 8: Export results
        results = analyzer.export_analysis_results()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("📊 Generated files:")
        print("   • sales_dashboard.png - Visual sales analysis")
        print("   • analysis_results.json - Detailed numeric results")
        print("\n🎯 Next steps:")
        print("   1. Review the generated visualizations")
        print("   2. Incorporate insights into your final report")
        print("   3. Use the JSON results for detailed metrics")
        
        return analyzer, insights, results
        
    else:
        print("❌ No data could be loaded. Please check your Excel files exist and are readable.")
        print("\n🔍 Troubleshooting steps:")
        print("   1. Verify all Excel files are in the same directory as this script")
        print("   2. Check file names match exactly:")
        for file in analyzer.data_files:
            print(f"      • {file}")
        print("   3. Ensure files are not password protected or corrupted")
        
        return None, None, None

# Execute if run directly
if __name__ == "__main__":
    analyzer, insights, results = run_comprehensive_analysis()
