import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import traceback
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################################################################
# NEWS SCRAPER CLASS (using free NewsAPI)
##################################################################
class NewsScraper:
    """
    Class to fetch news articles related to a company using NewsAPI.
    """
    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            logger.warning("NEWS_API_KEY not found in environment variables")

    def get_news_articles(self, company_name):
        """
        Fetch news articles for the given company name using NewsAPI.
        Returns a list of dictionaries, each containing a title and a URL.
        """
        articles = []
        try:
            if not self.api_key:
                logger.warning("Cannot fetch news articles: API key missing")
                return articles
                
            url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&language=en&apiKey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("articles", [])[:10]:  # Limit to 10 articles
                    articles.append({
                        "title": article.get("title"),
                        "description": article.get("description", ""),
                        "link": article.get("url"),
                        "publishedAt": article.get("publishedAt")
                    })
                logger.info(f"Retrieved {len(articles)} news articles for {company_name}")
            else:
                logger.error(f"News API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Exception in get_news_articles: {e}")
            traceback.print_exc()
        return articles

##################################################################
# SOCIAL MEDIA SCRAPER CLASS
##################################################################
class SocialMediaScraper:
    """
    Class to extract social media mentions for a company using DuckDuckGo search.
    """
    def get_social_media_mentions(self, company_name):
        """
        Scrape social media mentions for the given company name.
        Returns a list of dictionaries with content and URL.
        """
        mentions = []
        platforms = ["twitter", "linkedin", "facebook"]
        
        try:
            for platform in platforms:
                query = f"{company_name} {platform}"
                search_url = f"https://duckduckgo.com/html/?q={query}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                response = requests.get(search_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    results = soup.find_all('div', attrs={'class': 'result__body'})
                    
                    for i, result in enumerate(results):
                        if i >= 3:  # Limit to 3 results per platform
                            break
                            
                        title_elem = result.find('a', attrs={'class': 'result__a'})
                        snippet_elem = result.find('a', attrs={'class': 'result__snippet'})
                        
                        if title_elem and snippet_elem:
                            title = title_elem.get_text()
                            snippet = snippet_elem.get_text()
                            link = title_elem.get('href')
                            
                            mentions.append({
                                "platform": platform,
                                "title": title,
                                "content": snippet,
                                "link": link
                            })
                else:
                    logger.error(f"Social media search error for {platform}: {response.status_code}")
            
            logger.info(f"Retrieved {len(mentions)} social media mentions for {company_name}")
        except Exception as e:
            logger.error(f"SocialMediaScraper: Exception occurred: {e}")
            traceback.print_exc()
        
        return mentions

##################################################################
# THIRD-PARTY ESG RATINGS FETCHER CLASS
##################################################################
class ThirdPartyESGRatingsFetcher:
    """
    Class to fetch third-party ESG ratings for a company.
    """
    def __init__(self):
        self.api_key = os.getenv("ESG_API_KEY")
        if not self.api_key:
            logger.warning("ESG_API_KEY not found in environment variables")

    def get_esg_ratings(self, company_name):
        """
        Attempt to fetch real ESG ratings, with fallback to web scraping for public ESG information.
        """
        ratings = {}
        try:
            # Try to get ESG data from a public source
            search_url = f"https://www.google.com/search?q={company_name}+esg+rating+sustainability"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(search_url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract text containing potential ESG information
                paragraphs = soup.find_all(['p', 'span', 'div'])
                esg_text = ""
                
                for p in paragraphs:
                    text = p.get_text().lower()
                    if any(term in text for term in ['esg', 'environmental', 'social', 'governance', 'sustainability']):
                        esg_text += p.get_text() + " "
                
                # Return the collected ESG information
                if esg_text:
                    ratings["web_extracted_info"] = esg_text[:500]  # Limit length
                    logger.info(f"Extracted ESG information for {company_name}")
                else:
                    ratings["note"] = "No specific ESG ratings found through public web search"
            else:
                logger.error(f"ESG data search error: {response.status_code}")
                ratings["error"] = "Failed to retrieve ESG data"
        except Exception as e:
            logger.error(f"ThirdPartyESGRatingsFetcher: Exception occurred: {e}")
            traceback.print_exc()
            ratings["error"] = str(e)
        
        return ratings

##################################################################
# SUPPLIER INFO EXTRACTOR CLASS
##################################################################
class SupplierInfoExtractor:
    """
    Aggregates information from news, social media, and third-party ESG ratings.
    """
    def __init__(self):
        self.news_scraper = NewsScraper()
        self.social_scraper = SocialMediaScraper()
        self.esg_fetcher = ThirdPartyESGRatingsFetcher()

    def extract_supplier_info(self, company_name):
        """
        Extract supplier information from multiple sources.
        Returns a dictionary containing news articles, social media mentions, and ESG ratings.
        """
        supplier_info = {"company_name": company_name}
        
        # Add timestamp for when the analysis was performed
        supplier_info["analysis_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Get basic company info if possible
            supplier_info["search_term"] = company_name
            
            # Get news articles
            news_articles = self.news_scraper.get_news_articles(company_name)
            supplier_info["news_articles"] = news_articles
            supplier_info["news_count"] = len(news_articles)
            
            # Get social media mentions
            social_mentions = self.social_scraper.get_social_media_mentions(company_name)
            supplier_info["social_mentions"] = social_mentions
            supplier_info["social_mentions_count"] = len(social_mentions)
            
            # Get ESG ratings
            esg_ratings = self.esg_fetcher.get_esg_ratings(company_name)
            supplier_info["esg_ratings"] = esg_ratings
            
            logger.info(f"Completed information extraction for {company_name}")
        except Exception as e:
            logger.error(f"SupplierInfoExtractor: Exception occurred: {e}")
            traceback.print_exc()
            supplier_info["extraction_error"] = str(e)
        
        return supplier_info

##################################################################
# GROQ-BASED LLM CLASS
##################################################################
class GroqLLM(LLM):
    """
    Implementation of a Groq-based LLM using LangChain.
    Falls back to another provider if Groq isn't available.
    """
    temperature: float = 0.7
    max_tokens: int = 1500
    model_name: str = "groq-llm"
    api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Backup provider details
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt, stop=None):
        """
        Call the LLM API with a prompt, trying multiple providers if needed.
        """
        # Try Groq first if API key is available
        if self.api_key:
            try:
                logger.info("Attempting to use Groq API")
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama3-70b-8192",  # Or whatever model Groq supports
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                        headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Groq API call failed: {e}")
                traceback.print_exc()
        
        # Try OpenAI if API key is available
        if self.openai_api_key:
            try:
                logger.info("Attempting to use OpenAI API")
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                response = requests.post("https://api.openai.com/v1/chat/completions", 
                                        headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                traceback.print_exc()
        
        # Try Anthropic if API key is available
        if self.anthropic_api_key:
            try:
                logger.info("Attempting to use Anthropic API")
                headers = {
                    "x-api-key": self.anthropic_api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                response = requests.post("https://api.anthropic.com/v1/messages", 
                                        headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("content", [{}])[0].get("text", "")
                else:
                    logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Anthropic API call failed: {e}")
                traceback.print_exc()
        
        # Return an error message if all API calls fail
        logger.error("All LLM API calls failed. Check your API keys.")
        return "Error: Unable to connect to any LLM provider. Please check your API keys and try again."

    def predict(self, prompt):
        """
        Public method to get prediction from the LLM.
        """
        try:
            return self._call(prompt)
        except Exception as e:
            logger.error(f"LLM: Exception during prediction: {e}")
            traceback.print_exc()
            return "Error in processing the LLM request. See logs for details."

##################################################################
# REPUTATION ANALYZER AGENT CLASS
##################################################################
class ReputationAnalyzerAgent:
    """
    Agent that coordinates supplier info extraction and LLM-based reputation analysis.
    """
    def __init__(self):
        self.extractor = SupplierInfoExtractor()
        self.llm = GroqLLM()
        self.prompt_template = """
        You are an expert in supplier reputation analysis. Analyze the reputation of this supplier based on the following extracted information:

        Supplier Name: {company_name}
        Analysis Date: {timestamp}

        ### News Articles ({news_count} articles):
        {news_articles}

        ### Social Media Mentions ({social_count} mentions):
        {social_mentions}

        ### ESG Information:
        {esg_ratings}

        Provide a comprehensive reputation analysis with the following structure:
        1. Reputation Score: Give a score out of 10, with justification
        2. Key Strengths: List 3-5 positive aspects
        3. Key Concerns: List 3-5 negative aspects or areas of risk
        4. ESG Performance: Analyze environmental, social, and governance performance
        5. Media Sentiment: Analyze tone and sentiment of news coverage
        6. Social Media Perception: Analyze social media mentions
        7. Recommendations: Provide 3-5 recommendations for risk management

        Note: Be factual and balanced. Don't make assumptions beyond the data provided. If data is limited, acknowledge this limitation.
        """

    def analyze_reputation(self, company_name):
        """
        Extract supplier information and generate a reputation analysis using the LLM.
        """
        try:
            # Extract information
            logger.info(f"Beginning reputation analysis for {company_name}")
            info = self.extractor.extract_supplier_info(company_name)
            
            # Format news articles
            news_str = ""
            for i, article in enumerate(info.get("news_articles", [])[:5], 1):  # Limit to 5 articles for prompt
                news_str += f"{i}. Title: {article.get('title', 'No Title')}\n"
                news_str += f"   Description: {article.get('description', 'No Description')[:100]}...\n"
                news_str += f"   Published: {article.get('publishedAt', 'Unknown')}\n\n"
            
            # Format social media mentions
            social_str = ""
            for i, mention in enumerate(info.get("social_mentions", [])[:5], 1):  # Limit to 5 mentions
                social_str += f"{i}. Platform: {mention.get('platform', 'Unknown')}\n"
                social_str += f"   Content: {mention.get('content', 'No Content')[:100]}...\n\n"
            
            # Format ESG information
            esg_info = info.get("esg_ratings", {})
            if "web_extracted_info" in esg_info:
                esg_str = esg_info["web_extracted_info"]
            else:
                esg_str = "Limited ESG information available. Consider this a significant data gap."
            
            # Prepare the prompt
            prompt = self.prompt_template.format(
                company_name=company_name,
                timestamp=info.get("analysis_timestamp", "Unknown"),
                news_count=info.get("news_count", 0),
                news_articles=news_str if news_str else "No news articles found.",
                social_count=info.get("social_mentions_count", 0),
                social_mentions=social_str if social_str else "No social media mentions found.",
                esg_ratings=esg_str
            )
            
            logger.info("Sending analysis request to LLM")
            analysis = self.llm.predict(prompt)
            
            # Add data source summary
            sources_summary = f"""
            ----
            DATA SOURCES SUMMARY:
            - News Articles: {info.get('news_count', 0)} articles retrieved
            - Social Media: {info.get('social_mentions_count', 0)} mentions found
            - ESG Information: {"Available" if esg_str else "Not available"}
            - Analysis Date: {info.get('analysis_timestamp', 'Unknown')}
            """
            
            return analysis + sources_summary
        except Exception as e:
            logger.error(f"ReputationAnalyzerAgent: Exception occurred: {e}")
            traceback.print_exc()
            return f"Error in analyzing supplier reputation: {str(e)}"

##################################################################
# UTILS CLASS FOR HELPER FUNCTIONS
##################################################################
class Utils:
    """
    Utility class containing common helper functions.
    """
    @staticmethod
    def clean_text(text):
        """
        Clean and normalize text data by removing extra whitespace.
        """
        try:
            if not text:
                return ""
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.error(f"Utils.clean_text: Exception occurred: {e}")
            traceback.print_exc()
            return text

    @staticmethod
    def validate_url(url):
        """
        Validate if the URL is properly formatted.
        """
        try:
            pattern = re.compile(
                r'^(?:http|ftp)s?://'  # http:// or https://
                r'(?:[\w-]+\.)+[a-z]{2,6}'  # domain
                r'(?:/[\w.-]*)*/?$'
            )
            return bool(re.match(pattern, url))
        except Exception as e:
            logger.error(f"Utils.validate_url: Exception occurred: {e}")
            traceback.print_exc()
            return False
            
    @staticmethod
    def check_api_keys():
        """
        Check if required API keys are available and return status.
        """
        missing_keys = []
        
        if not os.getenv("NEWS_API_KEY"):
            missing_keys.append("NEWS_API_KEY")
            
        if not any([os.getenv("GROQ_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY")]):
            missing_keys.append("All LLM API keys (GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)")
            
        return missing_keys

##################################################################
# MAIN FUNCTION FOR STREAMLIT UI
##################################################################
def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Supplier Reputation Analyzer", layout="wide")
    st.title("Supplier Reputation Analysis Application")
    
    # Check API keys and display warnings if needed
    missing_keys = Utils.check_api_keys()
    if missing_keys:
        st.warning(f"Missing API keys: {', '.join(missing_keys)}. Some functionality may be limited.")
    
    st.write("Enter a company name to analyze its reputation based on news, social media, and ESG information.")
    
    # Input field for company name with example
    company_name = st.text_input("Company Name", placeholder="e.g., Microsoft, Tesla, Coca-Cola")
    
    # Analysis options
    with st.expander("Analysis Options", expanded=False):
        st.info("Advanced options will be added in future updates")
    
    analyze_button = st.button("Analyze Reputation")
    status_placeholder = st.empty()
    
    # Add sample companies for quick analysis
    st.markdown("#### Try these examples:")
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Analyze Microsoft"):
        company_name = "Microsoft"
        analyze_button = True
    
    if col2.button("Analyze Tesla"):
        company_name = "Tesla"
        analyze_button = True
    
    if col3.button("Analyze Coca-Cola"):
        company_name = "Coca-Cola"
        analyze_button = True
    
    if analyze_button:
        try:
            if company_name.strip() == "":
                st.error("Please enter a valid company name.")
            else:
                # Show analysis progress
                progress_bar = st.progress(0)
                status_placeholder.info("Starting reputation analysis...")
                
                # Create instance of ReputationAnalyzerAgent
                agent = ReputationAnalyzerAgent()
                
                # Update progress
                progress_bar.progress(25)
                status_placeholder.info("Gathering news articles and social media mentions...")
                
                # Run analysis
                progress_bar.progress(50)
                status_placeholder.info("Processing information and generating analysis...")
                
                result = agent.analyze_reputation(company_name)
                
                # Complete progress
                progress_bar.progress(100)
                status_placeholder.success("Analysis completed successfully!")
                
                # Display result in an expandable container with markdown formatting
                st.markdown("### Reputation Analysis Result")
                st.markdown(result)
                
                # Display timestamp
                st.caption(f"Analysis performed on {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.error(f"Main: Exception during analysis: {e}")
            traceback.print_exc()
            st.error("An error occurred during the analysis. Please try again later.")
            st.code(traceback.format_exc())

##################################################################
# APPLICATION ENTRY POINT
##################################################################
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application encountered a critical error: {e}")
        traceback.print_exc()
        st.error("Critical error occurred. Please check the logs for details.")