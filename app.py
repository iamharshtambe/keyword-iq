import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json

required_resources = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"
}

for resource, path in required_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

st.set_page_config(
    page_title="KeywordIQ - Advanced Text Analysis",
    page_icon="🧠",
    layout="wide"
)

# Enhanced Custom CSS
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
    .insight-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .entity-tag {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedKeywordExtractor:
    def __init__(self, text):
        self.text = text
        self.stop_words = set(stopwords.words('english'))
        self.sentences = sent_tokenize(text)
    
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
    
    def extract_trigrams(self, num_keywords=10):
        """Extract three-word phrases"""
        cleaned = self.clean_text(self.text)
        words = word_tokenize(cleaned)
        
        words = [word for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        trigrams = []
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            trigrams.append(trigram)
        
        trigram_freq = Counter(trigrams)
        return trigram_freq.most_common(num_keywords)
    
    def extract_named_entities(self):
        """Extract potential named entities using POS tagging"""
        words = word_tokenize(self.text)
        pos_tags = pos_tag(words)
        
        entities = []
        current_entity = []
        
        for word, tag in pos_tags:
            if tag in ['NNP', 'NNPS']:  # Proper nouns
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        return Counter(entities).most_common(15)
    
    def extract_pos_keywords(self, pos_filter=['NN', 'NNS', 'JJ', 'VB', 'VBG'], num_keywords=20):
        """Extract keywords based on part of speech"""
        words = word_tokenize(self.text.lower())
        pos_tags = pos_tag(words)
        
        filtered_words = [word for word, tag in pos_tags 
                         if tag in pos_filter and word not in self.stop_words and len(word) > 2]
        
        return Counter(filtered_words).most_common(num_keywords)
    
    def calculate_readability(self):
        """Calculate basic readability metrics"""
        words = word_tokenize(self.text)
        sentences = sent_tokenize(self.text)
        
        if not sentences or not words:
            return {}
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability score (0-100, higher is easier)
        readability_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 5)
        readability_score = max(0, min(100, readability_score))
        
        return {
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'readability_score': round(readability_score, 2)
        }
    
    def get_keyword_context(self, keyword, context_window=50):
        """Get context sentences for a keyword"""
        contexts = []
        keyword_lower = keyword.lower()
        
        for sentence in self.sentences[:10]:  # Limit to first 10 occurrences
            if keyword_lower in sentence.lower():
                contexts.append(sentence.strip())
        
        return contexts

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

def create_comparison_chart(freq_kw, tfidf_kw):
    """Create comparison chart between frequency and TF-IDF"""
    try:
        freq_dict = {word: score for word, score in freq_kw[:10]}
        tfidf_dict = {word: score for word, score in tfidf_kw[:10]}
        
        common_words = set(freq_dict.keys()) & set(tfidf_dict.keys())
        
        if not common_words:
            return None
        
        words = list(common_words)
        freq_scores = [freq_dict[w] for w in words]
        tfidf_scores = [tfidf_dict[w] * 100 for w in words]  # Scale TF-IDF
        
        fig = go.Figure(data=[
            go.Bar(name='Frequency', x=words, y=freq_scores),
            go.Bar(name='TF-IDF (scaled)', x=words, y=tfidf_scores)
        ])
        
        fig.update_layout(
            title='Keyword Comparison: Frequency vs TF-IDF',
            barmode='group',
            height=400
        )
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

def export_results(freq_kw, tfidf_kw, bigram_kw, stats):
    """Export results to JSON"""
    results = {
        'statistics': stats,
        'frequency_keywords': [{'keyword': w, 'score': s} for w, s in freq_kw],
        'tfidf_keywords': [{'keyword': w, 'score': float(s)} for w, s in tfidf_kw],
        'bigram_keywords': [{'keyword': w, 'score': s} for w, s in bigram_kw]
    }
    return json.dumps(results, indent=2)

def main():
    st.title("🧠 KeywordIQ - Advanced Text Analysis")
    st.markdown("### Extract keywords and analyze text using multiple NLP techniques")
    
    # Sidebar with enhanced settings
    st.sidebar.title("⚙️ Settings")
    num_keywords = st.sidebar.slider("Number of keywords", 5, 30, 15)
    
    st.sidebar.markdown("### 🎯 Analysis Options")
    show_entities = st.sidebar.checkbox("Show Named Entities", True)
    show_pos = st.sidebar.checkbox("Show POS-based Keywords", True)
    show_trigrams = st.sidebar.checkbox("Show Three-Word Phrases", True)
    show_readability = st.sidebar.checkbox("Show Readability Metrics", True)
    show_context = st.sidebar.checkbox("Show Keyword Context", False)
    
    # Main input area
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
            extractor = EnhancedKeywordExtractor(text_input)
            
            # Basic statistics
            words = len(text_input.split())
            chars = len(text_input)
            sentences = len(sent_tokenize(text_input))
            unique_words = len(set(word_tokenize(text_input.lower())))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", words)
            with col2:
                st.metric("Characters", chars)
            with col3:
                st.metric("Sentences", sentences)
            with col4:
                st.metric("Unique Words", unique_words)
            
            # Readability metrics
            if show_readability:
                readability = extractor.calculate_readability()
                if readability:
                    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                    st.markdown("### 📖 Readability Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Word Length", f"{readability['avg_word_length']} chars")
                    with col2:
                        st.metric("Avg Sentence Length", f"{readability['avg_sentence_length']} words")
                    with col3:
                        score = readability['readability_score']
                        difficulty = "Easy" if score > 60 else "Moderate" if score > 30 else "Difficult"
                        st.metric("Readability Score", f"{score}/100 ({difficulty})")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Extract all keywords
            freq_keywords = extractor.extract_frequency_keywords(num_keywords)
            tfidf_keywords = extractor.extract_tfidf_keywords(num_keywords)
            bigram_keywords = extractor.extract_bigram_keywords(num_keywords)
            
            # Create tabs for different views
            tabs = ["📋 Keywords", "📊 Visualizations", "🔍 Advanced Analysis"]
            tab1, tab2, tab3 = st.tabs(tabs)
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    display_keywords(freq_keywords, "Word Frequency")
                    st.markdown("<br>", unsafe_allow_html=True)
                    display_keywords(bigram_keywords, "Two-Word Phrases")
                
                with col2:
                    display_keywords(tfidf_keywords, "TF-IDF Keywords")
                    if show_trigrams:
                        st.markdown("<br>", unsafe_allow_html=True)
                        trigram_keywords = extractor.extract_trigrams(10)
                        display_keywords(trigram_keywords, "Three-Word Phrases")
            
            with tab2:
                st.subheader("📈 Keyword Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Word Cloud**")
                    wc_fig = create_wordcloud(freq_keywords)
                    if wc_fig:
                        st.pyplot(wc_fig)
                        plt.close()
                
                with col2:
                    st.write("**Top Keywords Chart**")
                    bar_fig = create_bar_chart(freq_keywords, "Most Frequent Words")
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)
                
                st.subheader("📊 TF-IDF Analysis")
                tfidf_fig = create_bar_chart(tfidf_keywords, "TF-IDF Scores")
                if tfidf_fig:
                    st.plotly_chart(tfidf_fig, use_container_width=True)
                
                st.subheader("🔄 Method Comparison")
                comp_fig = create_comparison_chart(freq_keywords, tfidf_keywords)
                if comp_fig:
                    st.plotly_chart(comp_fig, use_container_width=True)
            
            with tab3:
                if show_entities:
                    st.subheader("🏷️ Named Entities")
                    entities = extractor.extract_named_entities()
                    if entities:
                        tags_html = ""
                        for entity, count in entities:
                            tags_html += f'<span class="entity-tag">{entity} ({count})</span>'
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.write("No named entities found.")
                    st.markdown("<br>", unsafe_allow_html=True)
                
                if show_pos:
                    st.subheader("📝 Part-of-Speech Keywords")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Nouns**")
                        nouns = extractor.extract_pos_keywords(['NN', 'NNS'], 10)
                        display_keywords(nouns, "")
                    
                    with col2:
                        st.write("**Adjectives**")
                        adjectives = extractor.extract_pos_keywords(['JJ', 'JJR', 'JJS'], 10)
                        display_keywords(adjectives, "")
                
                if show_context and freq_keywords:
                    st.subheader("📄 Keyword Context")
                    selected_keyword = st.selectbox(
                        "Select a keyword to see its context:",
                        [kw for kw, _ in freq_keywords[:10]]
                    )
                    
                    if selected_keyword:
                        contexts = extractor.get_keyword_context(selected_keyword)
                        if contexts:
                            for i, context in enumerate(contexts[:3], 1):
                                st.markdown(f"**Context {i}:** {context}")
                        else:
                            st.write("No context found.")
            
            # Export functionality
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export to CSV
                df = pd.DataFrame({
                    'Frequency Keywords': [f"{w} ({s})" for w, s in freq_keywords],
                    'TF-IDF Keywords': [f"{w} ({s:.3f})" for w, s in tfidf_keywords] + [''] * (len(freq_keywords) - len(tfidf_keywords)),
                    'Bigram Keywords': [f"{w} ({s})" for w, s in bigram_keywords] + [''] * (len(freq_keywords) - len(bigram_keywords))
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="keywords_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export to JSON
                stats = {
                    'words': words,
                    'characters': chars,
                    'sentences': sentences,
                    'unique_words': unique_words
                }
                json_data = export_results(freq_keywords, tfidf_keywords, bigram_keywords, stats)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="keywords_analysis.json",
                    mime="application/json"
                )
            
            with col3:
                # Export to TXT
                txt_output = f"""KEYWORD ANALYSIS REPORT
{'='*50}

STATISTICS:
- Words: {words}
- Characters: {chars}
- Sentences: {sentences}
- Unique Words: {unique_words}

FREQUENCY KEYWORDS:
{chr(10).join([f"{i+1}. {w} ({s})" for i, (w, s) in enumerate(freq_keywords)])}

TF-IDF KEYWORDS:
{chr(10).join([f"{i+1}. {w} ({s:.3f})" for i, (w, s) in enumerate(tfidf_keywords)])}

BIGRAM KEYWORDS:
{chr(10).join([f"{i+1}. {w} ({s})" for i, (w, s) in enumerate(bigram_keywords)])}
"""
                st.download_button(
                    label="📥 Download TXT",
                    data=txt_output,
                    file_name="keywords_analysis.txt",
                    mime="text/plain"
                )
    
    st.markdown("---")
    st.markdown("""
    ### 🔬 About the Methods:
    - **Word Frequency**: Simple count of word occurrences
    - **TF-IDF**: Identifies statistically important words in the document
    - **N-grams**: Extracts common word sequences (bigrams and trigrams)
    - **Named Entities**: Identifies proper nouns and potential entity names
    - **POS Keywords**: Extracts keywords based on grammatical role
    - **Readability Metrics**: Analyzes text complexity and reading difficulty
    
    **KeywordIQ v2.0 - Created by: Harsh** | **NLP Mini Project**
    """)

if __name__ == "__main__":
    main()