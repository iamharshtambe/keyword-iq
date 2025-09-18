import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

required_resources = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords"
}

for resource, path in required_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


st.set_page_config(
    page_title="KeywordIQ - Simple Text Analysis",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .keyword-tag {
        display: inline-block;
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid #81d4fa;
    }
    .method-title {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleKeywordExtractor:
    def __init__(self, text):
        self.text = text
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def extract_frequency_keywords(self, num_keywords=20):
        """Extract keywords based on word frequency"""
        cleaned = self.clean_text(self.text)
        words = word_tokenize(cleaned)
        
        keywords = [word for word in words 
                   if word not in self.stop_words and len(word) > 2]
        
        word_freq = Counter(keywords)
        return word_freq.most_common(num_keywords)
    
    def extract_tfidf_keywords(self, num_keywords=20):
        """Extract keywords using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=num_keywords,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([self.text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            keywords = [(feature_names[i], tfidf_scores[i]) 
                       for i in range(len(feature_names)) if tfidf_scores[i] > 0]
            
            return sorted(keywords, key=lambda x: x[1], reverse=True)
        except:
            return []
    
    def extract_bigram_keywords(self, num_keywords=15):
        """Extract two-word phrases"""
        cleaned = self.clean_text(self.text)
        words = word_tokenize(cleaned)
        
        words = [word for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams.append(bigram)
        
        bigram_freq = Counter(bigrams)
        return bigram_freq.most_common(num_keywords)

def create_wordcloud(keywords):
    """Create word cloud from keywords"""
    if not keywords:
        return None
    
    freq_dict = {word: score for word, score in keywords[:30]}
    
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=30
        ).generate_from_frequencies(freq_dict)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except:
        return None

def create_bar_chart(keywords, title):
    """Create horizontal bar chart"""
    if not keywords:
        return None
    
    try:
        df = pd.DataFrame(keywords[:10], columns=['Keyword', 'Score'])
        
        fig = px.bar(
            df, 
            x='Score', 
            y='Keyword', 
            orientation='h',
            title=title,
            color='Score',
            color_continuous_scale='blues'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        return fig
    except:
        return None

def display_keywords(keywords, method_name):
    """Display keywords as tags"""
    st.markdown(f"<div class='method-title'>📊 {method_name}</div>", unsafe_allow_html=True)
    
    if not keywords:
        st.write("No keywords found.")
        return
    
    tags_html = ""
    for word, score in keywords[:15]:
        if isinstance(score, float):
            tags_html += f'<span class="keyword-tag">{word} ({score:.3f})</span>'
        else:
            tags_html += f'<span class="keyword-tag">{word} ({score})</span>'
    
    st.markdown(tags_html, unsafe_allow_html=True)

def main():
    st.title("🧠 KeywordIQ - Simple Text Analysis")
    st.markdown("### Extract keywords from text using simple but effective methods")
    
    st.sidebar.title("⚙️ Settings")
    num_keywords = st.sidebar.slider("Number of keywords", 5, 30, 15)
    
    st.markdown("### 📝 Enter Your Text")
    text_input = st.text_area(
        "Paste your text here:",
        height=200,
        placeholder="Enter the text you want to analyze for keywords..."
    )
    
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
    if uploaded_file:
        text_input = str(uploaded_file.read(), "utf-8")
        st.text_area("Uploaded content:", text_input, height=150, disabled=True)
    
    if st.button("🚀 Extract Keywords", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text first!")
            return
        
        with st.spinner("Analyzing text..."):
            extractor = SimpleKeywordExtractor(text_input)
            
            words = len(text_input.split())
            chars = len(text_input)
            sentences = len(sent_tokenize(text_input))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", words)
            with col2:
                st.metric("Characters", chars)
            with col3:
                st.metric("Sentences", sentences)
            
            st.markdown("---")
            
            freq_keywords = extractor.extract_frequency_keywords(num_keywords)
            tfidf_keywords = extractor.extract_tfidf_keywords(num_keywords)
            bigram_keywords = extractor.extract_bigram_keywords(num_keywords)
            
            tab1, tab2 = st.tabs(["📋 Keywords", "📊 Visualizations"])
            
            with tab1:
                display_keywords(freq_keywords, "Word Frequency")
                st.markdown("<br>", unsafe_allow_html=True)
                
                display_keywords(tfidf_keywords, "TF-IDF Keywords")
                st.markdown("<br>", unsafe_allow_html=True)
                
                display_keywords(bigram_keywords, "Two-Word Phrases")
            
            with tab2:
                st.subheader("Word Frequency Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Word Cloud**")
                    wc_fig = create_wordcloud(freq_keywords)
                    if wc_fig:
                        st.pyplot(wc_fig)
                        plt.close()
                    else:
                        st.write("Could not generate word cloud")
                
                with col2:
                    st.write("**Top Keywords Chart**")
                    bar_fig = create_bar_chart(freq_keywords, "Most Frequent Words")
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)
                    else:
                        st.write("Could not generate chart")
                
                st.subheader("TF-IDF Keywords")
                tfidf_fig = create_bar_chart(tfidf_keywords, "TF-IDF Scores")
                if tfidf_fig:
                    st.plotly_chart(tfidf_fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🔬 About the Methods:
    - **Word Frequency**: Simple count of how often words appear
    - **TF-IDF**: Term Frequency-Inverse Document Frequency - finds important words
    - **Two-Word Phrases**: Common word combinations in your text
    
    **KeywordIQ - Created by: Rushikesh, Harsh, Om** | **NLP Mini Project**
    """)

if __name__ == "__main__":
    main()