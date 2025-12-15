import requests
import pandas as pd
import nltk
import string

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Cleaning and Tokenizing the text
def preprocess_text(text):
    """Cleans and tokenizes the text."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return tokens

# API GDELT parameters
gdelt_base_url = 'https://api.gdeltproject.org/api/v2/doc/doc'
gdelt_queries = [
    'Trump shooting rally',
    'Trump assassination attempt',
    'Trump Pennsylvania rally 2024',
    'Trump July 2024 news',
    'Trump rally protest',
    'Trump Pennsylvania news',
    'Trump July 2024 incident',
    'Trump rally media coverage',
    'Trump security breach',
    'Trump rally attack',
    'Trump rally 2024 analysis',
    'Trump rally response 2024'
]

def collect_data_from_gdelt():
    """Collects data from the GDELT API for each specified query."""
    all_articles = pd.DataFrame()
    
    for query in gdelt_queries:
        params = {
            'query': query,
            'mode': 'ArtList',
            'maxrecords': 250,
            'format': 'json'
        }
        try:
            response = requests.get(gdelt_base_url, params=params)
            response.raise_for_status()  # Check for any HTTP errors
            data = response.json()
            if 'articles' in data:
                articles = pd.DataFrame(data['articles'])
                all_articles = pd.concat([all_articles, articles], ignore_index=True)
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} for query '{query}'")
        except Exception as err:
            print(f"Other error occurred: {err} for query '{query}'")
    
    return all_articles

def combine_and_clean_data(new_data):
    """Combines and cleans the new data with the existing dataset."""
    try:
        existing_data = pd.read_csv('/path/to/your/dataset/combined_trump_data_cleaned.csv')
    except FileNotFoundError:
        existing_data = pd.DataFrame()
    
    # Combining with existing data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Removing duplicates by 'url'
    combined_data.drop_duplicates(subset='url', inplace=True)
    
    # Cleaning the data by removing missing rows
    combined_data.dropna(subset=['url', 'title', 'content'], inplace=True)
    
    # Save the dataset
    combined_data.to_csv('/path/to/your/dataset/combined_trump_data_cleaned.csv', index=False)
    print(f"Combined data saved to combined_trump_data_cleaned.csv with {combined_data.shape[0]} total articles.")
    
    return combined_data

# Main function to collect and process data
def main():
    """Main function to execute data collection and combination."""
    gdelt_data = collect_data_from_gdelt()

    if not gdelt_data.empty:
        combined_data = combine_and_clean_data(gdelt_data)
        print(f"Data combined and cleaned. Total articles after combining: {combined_data.shape[0]}")
    else:
        print("No new articles collected from GDELT.")

if __name__ == "__main__":
    main()
