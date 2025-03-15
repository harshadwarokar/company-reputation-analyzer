# company-reputation-analyzer
A reputation analysis tool that gathers news, social media mentions, and ESG ratings for companies using APIs and web scraping. It leverages LLMs (Groq, OpenAI, Anthropic) for sentiment analysis via a Streamlit UI.

## Overview
This project is a **Company Reputation Analysis Application** that extracts and analyzes a company's reputation based on news articles, social media mentions, and ESG (Environmental, Social, and Governance) ratings. The tool leverages **NewsAPI**, **DuckDuckGo search**, and web scraping techniques to gather relevant data. Additionally, it uses **LLM-based reputation analysis** via **Groq**, **OpenAI**, or **Anthropic** APIs.

## Features
- **News Scraper**: Fetches company-related news articles using **NewsAPI**.
- **Social Media Scraper**: Extracts mentions from **Twitter, LinkedIn, and Facebook** via **DuckDuckGo** search.
- **ESG Ratings Fetcher**: Retrieves ESG ratings via Google search.
- **LLM-Based Reputation Analysis**: Uses **Groq**, **OpenAI**, or **Anthropic** models for sentiment and risk assessment.
- **Streamlit UI**: Provides an interactive web interface for reputation analysis.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/company-reputation-analyzer.git
   cd company-reputation-analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys:
   - Create a `.env` file and add the following:
     ```env
     NEWS_API_KEY="your_news_api_key"
     GROQ_API_KEY="your_groq_api_key"
     OPENAI_API_KEY="your_openai_api_key"  # Optional
     ANTHROPIC_API_KEY="your_anthropic_api_key"  # Optional
     ```

## Usage
### Running the Application
To launch the **Streamlit UI**, run:
```bash
streamlit run reputation.py
```

### Analyzing a Company
1. Enter the company name in the input field.
2. Click **Analyze Reputation**.
3. View extracted **news articles, social media mentions, and ESG insights**.
4. Read the AI-generated **reputation analysis**.

## Technologies Used
- **Python** (FastAPI, Requests, BeautifulSoup, dotenv)
- **Streamlit** (UI Framework)
- **LLMs** (Groq, OpenAI, Anthropic API)
- **NewsAPI** (News data source)
- **DuckDuckGo** (Social media search)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
1. Fork the repository.
2. Create a new branch (`feature-xyz`).
3. Commit your changes.
4. Submit a pull request.

## Contact
For any questions, contact Harshad warokar at **harshadwarokar@gmail.com** or create an issue in this repository.

