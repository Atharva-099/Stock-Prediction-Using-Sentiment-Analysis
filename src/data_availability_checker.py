"""
Data Availability Checker for Financial News
=============================================
Checks what years of data are available from:
1. HuggingFace datasets (FNSPID 1999-2023, Multi-source 1990-2025)
2. Google RSS feeds (recent data only)
3. Other potential sources

Author: CMU Financial Forecasting Project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class DataAvailabilityChecker:
    """Check and report data availability across sources"""
    
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.availability_report = {}
        
    def check_huggingface_fnspid(self, sample_size=50000):
        """
        Check FNSPID dataset (1999-2023) availability
        
        The FNSPID (Financial News and Stock Price Integration Dataset) contains:
        - 29.7 million stock prices
        - 15.7 million time-aligned financial news records
        - Covers 4,775 S&P 500 companies
        - Date range: 1999-2023
        """
        from datasets import load_dataset
        
        logger.info("=" * 80)
        logger.info("CHECKING HUGGINGFACE FNSPID DATASET (1999-2023)")
        logger.info("=" * 80)
        
        dataset_name = "Brianferrell787/financial-news-multisource"
        data_files = "data/fnspid_news/*.parquet"
        
        yearly_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        date_range = {'min': None, 'max': None}
        sources = defaultdict(int)
        
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            logger.info(f"Data files: {data_files}")
            
            ds = load_dataset(
                dataset_name,
                data_files=data_files,
                split="train",
                streaming=True,
                token=self.hf_token
            )
            
            logger.info(f"Scanning {sample_size:,} records to determine availability...")
            
            count = 0
            for row in tqdm(ds, desc="FNSPID Scan", total=sample_size):
                if count >= sample_size:
                    break
                    
                try:
                    article_date = pd.to_datetime(row['date'])
                    year = article_date.year
                    month_key = f"{year}-{article_date.month:02d}"
                    
                    yearly_counts[year] += 1
                    monthly_counts[month_key] += 1
                    
                    if date_range['min'] is None or article_date < date_range['min']:
                        date_range['min'] = article_date
                    if date_range['max'] is None or article_date > date_range['max']:
                        date_range['max'] = article_date
                    
                    # Extract source
                    try:
                        extra = json.loads(row.get('extra_fields', '{}'))
                        source = extra.get('source', 'Unknown')
                        sources[source] += 1
                    except:
                        sources['Unknown'] += 1
                        
                    count += 1
                except Exception as e:
                    continue
            
            # Compile report
            fnspid_report = {
                'dataset': 'FNSPID (Financial News and Stock Price Integration)',
                'source': 'Brianferrell787/financial-news-multisource',
                'data_files': data_files,
                'expected_range': '1999-2023',
                'scanned_records': count,
                'found_date_range': {
                    'min': str(date_range['min']) if date_range['min'] else None,
                    'max': str(date_range['max']) if date_range['max'] else None
                },
                'yearly_distribution': dict(sorted(yearly_counts.items())),
                'top_sources': dict(sorted(sources.items(), key=lambda x: -x[1])[:10]),
                'years_available': sorted(yearly_counts.keys())
            }
            
            self.availability_report['fnspid'] = fnspid_report
            
            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("FNSPID SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Date Range Found: {date_range['min']} to {date_range['max']}")
            logger.info(f"Years Available: {sorted(yearly_counts.keys())}")
            logger.info("\nYearly Distribution (from sample):")
            for year in sorted(yearly_counts.keys()):
                logger.info(f"  {year}: {yearly_counts[year]:,} articles")
            
            return fnspid_report
            
        except Exception as e:
            logger.error(f"Error checking FNSPID: {e}")
            return {'error': str(e)}
    
    def check_huggingface_multisource(self, sample_size=50000):
        """
        Check the full multi-source dataset
        
        The Multi-Source Financial & General News dataset contains:
        - 57+ million news articles
        - Multiple source datasets aggregated
        - Date range: 1990-2025 (across all sources)
        """
        from datasets import load_dataset
        
        logger.info("\n" + "=" * 80)
        logger.info("CHECKING HUGGINGFACE MULTI-SOURCE DATASET (1990-2025)")
        logger.info("=" * 80)
        
        dataset_name = "Brianferrell787/financial-news-multisource"
        
        # Check different subsets
        subsets = [
            ("data/fnspid_news/*.parquet", "FNSPID News (1999-2023)"),
            ("data/reuters_news/*.parquet", "Reuters News"),
            ("data/cnbc_headlines/*.parquet", "CNBC Headlines"),
            ("data/seeking_alpha/*.parquet", "Seeking Alpha"),
            ("data/benzinga/*.parquet", "Benzinga"),
            ("data/cnn_news/*.parquet", "CNN News"),
            ("data/bbc_news/*.parquet", "BBC News"),
            ("data/nyt_news/*.parquet", "NYT News"),
        ]
        
        all_years = set()
        subset_reports = {}
        
        for data_files, subset_name in subsets:
            logger.info(f"\nChecking: {subset_name}")
            yearly_counts = defaultdict(int)
            
            try:
                ds = load_dataset(
                    dataset_name,
                    data_files=data_files,
                    split="train",
                    streaming=True,
                    token=self.hf_token
                )
                
                count = 0
                for row in tqdm(ds, desc=subset_name[:20], total=sample_size // len(subsets)):
                    if count >= sample_size // len(subsets):
                        break
                    try:
                        article_date = pd.to_datetime(row['date'])
                        yearly_counts[article_date.year] += 1
                        all_years.add(article_date.year)
                        count += 1
                    except:
                        continue
                
                subset_reports[subset_name] = {
                    'years': sorted(yearly_counts.keys()),
                    'counts': dict(sorted(yearly_counts.items())),
                    'total_sampled': count
                }
                
                logger.info(f"  Years: {sorted(yearly_counts.keys())}")
                
            except Exception as e:
                logger.warning(f"  Could not access: {e}")
                subset_reports[subset_name] = {'error': str(e)}
        
        multisource_report = {
            'dataset': 'Multi-Source Financial News',
            'source': 'Brianferrell787/financial-news-multisource',
            'subsets_checked': subset_reports,
            'all_years_found': sorted(all_years),
            'date_range': f"{min(all_years) if all_years else 'N/A'} - {max(all_years) if all_years else 'N/A'}"
        }
        
        self.availability_report['multisource'] = multisource_report
        return multisource_report
    
    def check_ashraq_financial_news(self, sample_size=20000):
        """
        Check ashraq/financial-news dataset (2020-2025)
        """
        from datasets import load_dataset
        
        logger.info("\n" + "=" * 80)
        logger.info("CHECKING ASHRAQ/FINANCIAL-NEWS DATASET (2020-2025)")
        logger.info("=" * 80)
        
        yearly_counts = defaultdict(int)
        date_range = {'min': None, 'max': None}
        
        try:
            ds = load_dataset(
                "ashraq/financial-news",
                split="train",
                streaming=True
            )
            
            count = 0
            for row in tqdm(ds, desc="ashraq/financial-news", total=sample_size):
                if count >= sample_size:
                    break
                try:
                    # Try different date field names
                    date_str = row.get('date') or row.get('published_date') or row.get('timestamp')
                    if date_str:
                        article_date = pd.to_datetime(date_str)
                        yearly_counts[article_date.year] += 1
                        
                        if date_range['min'] is None or article_date < date_range['min']:
                            date_range['min'] = article_date
                        if date_range['max'] is None or article_date > date_range['max']:
                            date_range['max'] = article_date
                    count += 1
                except:
                    continue
            
            report = {
                'dataset': 'ashraq/financial-news',
                'expected_range': '2020-2025',
                'found_date_range': {
                    'min': str(date_range['min']) if date_range['min'] else None,
                    'max': str(date_range['max']) if date_range['max'] else None
                },
                'yearly_distribution': dict(sorted(yearly_counts.items())),
                'years_available': sorted(yearly_counts.keys()),
                'sampled': count
            }
            
            self.availability_report['ashraq'] = report
            
            logger.info(f"Date Range: {date_range['min']} to {date_range['max']}")
            logger.info(f"Years: {sorted(yearly_counts.keys())}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error checking ashraq dataset: {e}")
            return {'error': str(e)}
    
    def check_rss_availability(self):
        """
        Check what data is available via RSS feeds
        RSS feeds typically only provide recent data (last few days/weeks)
        """
        import feedparser
        
        logger.info("\n" + "=" * 80)
        logger.info("CHECKING RSS FEED AVAILABILITY")
        logger.info("=" * 80)
        
        rss_sources = {
            'Google News (Finance)': 'https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en',
            'Yahoo Finance': 'https://finance.yahoo.com/rss/',
            'CNBC': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147',
            'MarketWatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        }
        
        rss_report = {}
        
        for source_name, url in rss_sources.items():
            try:
                logger.info(f"Checking: {source_name}")
                feed = feedparser.parse(url)
                
                dates = []
                for entry in feed.entries[:50]:
                    try:
                        pub_date = pd.to_datetime(entry.published)
                        dates.append(pub_date)
                    except:
                        continue
                
                if dates:
                    rss_report[source_name] = {
                        'url': url,
                        'articles_available': len(feed.entries),
                        'date_range': {
                            'min': str(min(dates)),
                            'max': str(max(dates))
                        },
                        'days_of_coverage': (max(dates) - min(dates)).days
                    }
                    logger.info(f"  ✓ {len(feed.entries)} articles, {min(dates).date()} to {max(dates).date()}")
                else:
                    rss_report[source_name] = {'error': 'No valid dates found'}
                    
            except Exception as e:
                rss_report[source_name] = {'error': str(e)}
                logger.warning(f"  ✗ Error: {e}")
        
        self.availability_report['rss'] = rss_report
        return rss_report
    
    def generate_full_report(self, sample_size=30000):
        """Generate comprehensive availability report"""
        logger.info("\n" + "=" * 100)
        logger.info(" DATA AVAILABILITY REPORT - FINANCIAL NEWS (1999-2025)")
        logger.info("=" * 100)
        
        # Check all sources
        self.check_huggingface_fnspid(sample_size)
        self.check_ashraq_financial_news(sample_size // 2)
        self.check_rss_availability()
        
        # Generate summary
        logger.info("\n" + "=" * 100)
        logger.info(" SUMMARY: DATA AVAILABILITY BY YEAR")
        logger.info("=" * 100)
        
        # Collect all years
        year_sources = defaultdict(list)
        
        if 'fnspid' in self.availability_report and 'years_available' in self.availability_report['fnspid']:
            for year in self.availability_report['fnspid']['years_available']:
                year_sources[year].append('FNSPID (HuggingFace)')
        
        if 'ashraq' in self.availability_report and 'years_available' in self.availability_report['ashraq']:
            for year in self.availability_report['ashraq']['years_available']:
                year_sources[year].append('ashraq/financial-news')
        
        # RSS is always current year
        current_year = datetime.now().year
        year_sources[current_year].append('RSS Feeds')
        year_sources[current_year - 1].append('RSS Feeds (partial)')
        
        logger.info("\nYear-by-Year Availability:")
        logger.info("-" * 60)
        for year in sorted(year_sources.keys()):
            sources = year_sources[year]
            logger.info(f"  {year}: {', '.join(sources)}")
        
        # Training/Fine-tuning recommendations
        all_years = sorted(year_sources.keys())
        if all_years:
            train_years = [y for y in all_years if y <= 2022]
            finetune_years = [y for y in all_years if y >= 2023]
            
            logger.info("\n" + "=" * 60)
            logger.info("RECOMMENDED TRAINING STRATEGY")
            logger.info("=" * 60)
            logger.info(f"\n📚 TRAINING DATA (Historical):")
            logger.info(f"   Years: {min(train_years) if train_years else 'N/A'} - {max(train_years) if train_years else 'N/A'}")
            logger.info(f"   Source: FNSPID Dataset (HuggingFace)")
            logger.info(f"\n🔧 FINE-TUNING DATA (Recent):")
            logger.info(f"   Years: {min(finetune_years) if finetune_years else 'N/A'} - {max(finetune_years) if finetune_years else 'N/A'}")
            logger.info(f"   Sources: RSS Feeds + ashraq/financial-news")
            logger.info(f"\n✅ VALIDATION DATA (Current):")
            logger.info(f"   Year: {current_year}")
            logger.info(f"   Source: RSS Feeds (real-time)")
        
        self.availability_report['summary'] = {
            'all_years': all_years,
            'year_sources': dict(year_sources),
            'train_years': train_years if all_years else [],
            'finetune_years': finetune_years if all_years else [],
            'validation_year': current_year
        }
        
        return self.availability_report
    
    def save_report(self, filepath='results/data_availability_report.json'):
        """Save report to file"""
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert any remaining datetime objects
        def convert_dates(obj):
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(i) for i in obj]
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return str(obj)
            else:
                return obj
        
        report = convert_dates(self.availability_report)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n✓ Report saved to: {filepath}")
        return filepath


if __name__ == "__main__":
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get HF token
    hf_token = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')
    
    checker = DataAvailabilityChecker(hf_token=hf_token)
    report = checker.generate_full_report(sample_size=30000)
    checker.save_report()


