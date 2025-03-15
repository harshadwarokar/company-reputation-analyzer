from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import logging
import traceback
import re
import os
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Supplier Reputation API", 
              description="API for analyzing supplier reputation")

##################################################################
# PYDANTIC MODELS
##################################################################

class CompanyRequest(BaseModel):
    company_name: str

class ReputationResponse(BaseModel):
    company_name: str
    analysis: str
    timestamp: str
    data_sources: Dict[str, Any]
    status: str

##################################################################
# NEWS SCRAPER CLASS
##################################################################
class NewsScraper:
    """Class to fetch news articles related to a company using NewsAPI."""
    
    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            logger.warning("NEWS_API_KEY not found in environment variables")

    def get_news_articles(self, company_name: str) -> List[Dict]:
        """Fetch news articles for the given company name using NewsAPI."""
        articles = []
        try:
            if not self.api_key:
                logger.warning("Cannot fetch news articles: API key missing")
                return articles
                
            url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&language=en&apiKey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("articles", [])[:8]:  # Limit to 8 articles
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
    """Class to extract social media mentions for a company."""
    
    def get_social_media_mentions(self, company_name: str) -> List[Dict]:
        """Scrape social media mentions for the given company name."""
        mentions = []
        platforms = ["twitter", "linkedin"]
        
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
    """Class to fetch ESG-related information for a company."""
    
    def get_esg_ratings(self, company_name: str) -> Dict:
        """Attempt to fetch ESG information via web search."""
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
    """Aggregates information from news, social media, and ESG sources."""
    
    def __init__(self):
        self.news_scraper = NewsScraper()
        self.social_scraper = SocialMediaScraper()
        self.esg_fetcher = ThirdPartyESGRatingsFetcher()

    def extract_supplier_info(self, company_name: str) -> Dict:
        """Extract supplier information from multiple sources."""
        supplier_info = {"company_name": company_name}
        
        # Add timestamp for when the analysis was performed
        supplier_info["analysis_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
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
# GROQ LLM CLASS
##################################################################
class GroqLLM:
    """Implementation of a Groq-based LLM client."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        self.temperature = 0.7
        self.max_tokens = 1500

    def analyze(self, prompt: str) -> str:
        """Call the Groq API with a prompt and return the response."""
        if not self.api_key:
            return "Error: GROQ_API_KEY missing. Cannot perform analysis."
            
        try:
            logger.info("Calling Groq API for analysis")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-70b-8192",  # Or another Groq model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error: Groq API returned status code {response.status_code}"
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            traceback.print_exc()
            return f"Error in processing the LLM request: {str(e)}"

##################################################################
# REPUTATION ANALYZER CLASS
##################################################################
class ReputationAnalyzer:
    """Coordinates supplier info extraction and LLM-based reputation analysis."""
    
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

    def analyze_reputation(self, company_name: str) -> Dict:
        """Extract supplier information and generate a reputation analysis."""
        try:
            # Extract information
            logger.info(f"Beginning reputation analysis for {company_name}")
            info = self.extractor.extract_supplier_info(company_name)
            
            # Format news articles
            news_str = ""
            for i, article in enumerate(info.get("news_articles", [])[:5], 1):
                news_str += f"{i}. Title: {article.get('title', 'No Title')}\n"
                news_str += f"   Description: {article.get('description', 'No Description')[:100]}...\n"
                news_str += f"   Published: {article.get('publishedAt', 'Unknown')}\n\n"
            
            # Format social media mentions
            social_str = ""
            for i, mention in enumerate(info.get("social_mentions", [])[:5], 1):
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
            
            # Get analysis from LLM
            logger.info("Sending analysis request to Groq LLM")
            analysis = self.llm.analyze(prompt)
            
            # Return results
            return {
                "company_name": company_name,
                "analysis": analysis,
                "timestamp": info.get("analysis_timestamp", "Unknown"),
                "data_sources": {
                    "news_count": info.get("news_count", 0),
                    "social_mentions_count": info.get("social_mentions_count", 0),
                    "esg_info_available": "web_extracted_info" in esg_info
                },
                "status": "success"
            }
        except Exception as e:
            logger.error(f"ReputationAnalyzer: Exception occurred: {e}")
            traceback.print_exc()
            return {
                "company_name": company_name,
                "analysis": f"Error in analyzing supplier reputation: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_sources": {},
                "status": "error"
            }

##################################################################
# API ENDPOINTS
##################################################################

# Create analyzer instance
reputation_analyzer = ReputationAnalyzer()

# Define the single endpoint
@app.post("/analyze", response_model=ReputationResponse)
async def analyze_company(request: CompanyRequest, background_tasks: BackgroundTasks):
    """
    Analyze a company's reputation based on news, social media, and ESG information.
    
    Takes a company name and returns a reputation analysis.
    """
    company_name = request.company_name.strip()
    
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name must not be empty")
    
    # Process the request in a background task
    result = reputation_analyzer.analyze_reputation(company_name)
    
    return result

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Supplier Reputation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)