# ============================================================
# STREAMLIT FRONTEND FOR DISCOURSE FEATURE ANALYZER
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time

# Configuration
API_URL = "http://127.0.0.1:8000/extract-features"
API_HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="Discourse Feature Analyzer", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3D58;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #1E3D58;
    }
    .feature-category {
        font-weight: bold;
        color: #1E3D58;
        font-size: 1.2rem;
        margin-top: 20px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #1E3D58;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3D58;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=60)
def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(API_HEALTH_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def categorize_features(features_dict):
    """Categorize features by their prefix"""
    categories = defaultdict(dict)
    
    for key, value in features_dict.items():
        if '.' in key:
            category = key.split('.')[0]
        else:
            # Handle Coh-Metrix features (no dots)
            if key[:2].isupper() and len(key) > 2:  # Coh-Metrix feature codes like DESWC
                category = "cohmetrix"
            else:
                category = "other"
        
        categories[category][key] = value
    
    return categories

def plot_feature_category(category_features, category_name, top_n=15):
    """Create a bar plot for features in a category"""
    if not category_features:
        return None
    
    # Sort by value
    sorted_items = sorted(
        category_features.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:top_n]
    
    names = [item[0].split('.')[-1] if '.' in item[0] else item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Color based on positive/negative
    colors = ['#2E86AB' if v > 0 else '#A23B72' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        title=f"{category_name} Features",
        xaxis_title="Value",
        yaxis_title="Feature",
        height=max(400, len(names) * 25),
        margin=dict(l=150, r=50, t=50, b=50),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_readability_metrics(features):
    """Create visualizations for readability metrics"""
    readability_features = {}
    
    # Extract readability formulas
    for key, value in features.items():
        if key.startswith('formulas.'):
            short_name = key.replace('formulas.', '')
            readability_features[short_name] = value
    
    if not readability_features:
        return None
    
    # Create two columns for different visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of readability scores
        fig1 = go.Figure(data=[
            go.Bar(
                x=list(readability_features.values()),
                y=list(readability_features.keys()),
                orientation='h',
                marker_color='#2E86AB',
                text=[f"{v:.2f}" for v in readability_features.values()],
                textposition='outside'
            )
        ])
        fig1.update_layout(
            title="Readability Scores",
            xaxis_title="Score",
            yaxis_title="Formula",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Gauge chart for Flesch Reading Ease
        if 'flesch_reading_ease' in readability_features:
            fre_score = readability_features['flesch_reading_ease']
            
            # Determine grade level based on Flesch score
            if fre_score >= 90:
                grade = "5th grade"
                color = "green"
            elif fre_score >= 80:
                grade = "6th grade"
                color = "lightgreen"
            elif fre_score >= 70:
                grade = "7th grade"
                color = "yellowgreen"
            elif fre_score >= 60:
                grade = "8th-9th grade"
                color = "gold"
            elif fre_score >= 50:
                grade = "10th-12th grade"
                color = "orange"
            elif fre_score >= 30:
                grade = "College"
                color = "orangered"
            else:
                grade = "College Graduate"
                color = "red"
            
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fre_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Flesch Reading Ease<br><span style='font-size:0.8em;color:{color}'>{grade}</span>"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "yellow"},
                        {'range': [60, 70], 'color': "yellowgreen"},
                        {'range': [70, 80], 'color': "lightgreen"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

def plot_entity_analysis(entity_features):
    """Create visualizations for entity-related features"""
    if not entity_features:
        return None
    
    # Extract relevant entity metrics
    metrics = {
        'Entity Density': entity_features.get('ent.entity_density', 0),
        'Named Entity Density': entity_features.get('ent.named_entity_density', 0),
        'Unique Entity Density': entity_features.get('ent.unique_entity_density', 0),
        'Named Entity %': entity_features.get('ent.percent_named_entities', 0),
        'General Noun %': entity_features.get('ent.percent_general_nouns', 0)
    }
    
    # Create pie chart for entity composition
    if 'ent.percent_named_entities' in entity_features and 'ent.percent_general_nouns' in entity_features:
        fig1 = go.Figure(data=[go.Pie(
            labels=['Named Entities', 'General Nouns'],
            values=[
                entity_features['ent.percent_named_entities'],
                entity_features['ent.percent_general_nouns']
            ],
            hole=.3,
            marker_colors=['#2E86AB', '#A23B72']
        )])
        fig1.update_layout(
            title="Entity Type Distribution",
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Create bar chart for entity metrics
    fig2 = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#2E86AB', '#A23B72', '#F18F01', '#4ECDC4', '#96CEB4'],
            text=[f"{v:.2f}" for v in metrics.values()],
            textposition='outside'
        )
    ])
    fig2.update_layout(
        title="Entity Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=350,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig2, use_container_width=True)

def plot_discourse_structure(coref_features, lex_features, const_features):
    """Create visualizations for discourse structure features"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Coreference features
        if coref_features:
            coref_metrics = {
                'Num Chains': coref_features.get('coref.num_chains', 0),
                'Avg Mentions/Chain': coref_features.get('coref.avg_refs_per_chain', 0),
                'Avg Chain Span': coref_features.get('coref.avg_chain_span', 0),
                'Long Span Chains': coref_features.get('coref.long_span_chains', 0)
            }
            
            fig1 = go.Figure(data=[
                go.Bar(
                    x=list(coref_metrics.values()),
                    y=list(coref_metrics.keys()),
                    orientation='h',
                    marker_color='#2E86AB',
                    text=[f"{v:.2f}" for v in coref_metrics.values()],
                    textposition='outside'
                )
            ])
            fig1.update_layout(
                title="Coreference Resolution",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Lexical chain features
        if lex_features:
            lex_metrics = {
                'Num Chains': lex_features.get('lex.num_lexical_chains', 0),
                'Avg Length': lex_features.get('lex.avg_chain_length', 0),
                'Avg Span': lex_features.get('lex.avg_chain_span', 0),
                'Long Span Chains': lex_features.get('lex.long_span_chains', 0)
            }
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=list(lex_metrics.values()),
                    y=list(lex_metrics.keys()),
                    orientation='h',
                    marker_color='#A23B72',
                    text=[f"{v:.2f}" for v in lex_metrics.values()],
                    textposition='outside'
                )
            ])
            fig2.update_layout(
                title="Lexical Chains",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Constituency features in full width
    if const_features:
        const_metrics = {
            'VP Density': const_features.get('const.vp_density', 0),
            'NP Density': const_features.get('const.np_density', 0),
            'ADJP Density': const_features.get('const.adjp_density', 0),
            'ADVP Density': const_features.get('const.advp_density', 0),
            'SBAR Density': const_features.get('const.sbar_density', 0),
            'Avg Tree Height': const_features.get('const.avg_tree_height', 0)
        }
        
        fig3 = go.Figure(data=[
            go.Bar(
                x=list(const_metrics.keys()),
                y=list(const_metrics.values()),
                marker_color='#F18F01',
                text=[f"{v:.3f}" for v in const_metrics.values()],
                textposition='outside'
            )
        ])
        fig3.update_layout(
            title="Constituency Parsing",
            xaxis_title="Feature",
            yaxis_title="Value",
            height=350,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig3, use_container_width=True)

def plot_pos_analysis(pos_features):
    """Create visualizations for POS features"""
    if not pos_features:
        return None
    
    # Create radar chart for POS distribution
    categories = ['Nouns', 'Prepositions', 'Adjectives', 'Adverbs', '1st Person', '3rd Person']
    
    values = [
        pos_features.get('pos.noun_density', 0),
        pos_features.get('pos.preposition_density', 0),
        pos_features.get('pos.adjective_density', 0),
        pos_features.get('pos.adverb_density', 0),
        pos_features.get('pos.first_person_pronoun_density', 0),
        pos_features.get('pos.third_person_pronoun_density', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker_color='#2E86AB',
        line_color='#1E3D58'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2 if values else 10]
            )),
        showlegend=False,
        title="Part-of-Speech Distribution (per 100 words)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_entity_grid_transitions(eg_features):
    """Create heatmap of entity grid transitions"""
    if not eg_features:
        return None
    
    # Extract transition probabilities
    roles = ['S', 'O', 'X', '-']
    transition_matrix = np.zeros((4, 4))
    
    for i, r1 in enumerate(roles):
        for j, r2 in enumerate(roles):
            key = f"eg.trans_{r1}_{r2}"
            transition_matrix[i, j] = eg_features.get(key, 0)
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=roles,
        y=roles,
        colorscale='Blues',
        text=[[f"{val:.3f}" for val in row] for row in transition_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title="Entity Grid Role Transitions",
        xaxis_title="To Role",
        yaxis_title="From Role",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Advanced Discourse Feature Analyzer</h1>', unsafe_allow_html=True)
    
    # Check API health
    health_status = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/linguistics.png", width=80)
        st.header("🔧 Configuration")
        
        # API Status
        st.subheader("📡 API Status")
        if health_status:
            st.success("✅ Backend Connected")
            if health_status.get('models_loaded'):
                models = health_status['models_loaded']
                for model, loaded in models.items():
                    if loaded:
                        st.info(f"✓ {model.title()}")
            if health_status.get('cohmetrix_available'):
                st.success("📊 Coh-Metrix Available")
        else:
            st.error("❌ Backend Not Connected")
            st.markdown("""
            **To start the backend:**
            ```bash
            uvicorn app:app --reload
            ```
            """)
        
        st.divider()
        
        # Analysis Options
        st.subheader("⚙️ Analysis Options")
        include_lftk = st.checkbox("Include LFTK Features", value=True, 
                                   help="Language Feature Toolkit - additional linguistic measures")
        include_cohmetrix = st.checkbox("Include Coh-Metrix", value=False,
                                       help="Coh-Metrix cohesion metrics (requires CLI)")
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This analyzer extracts **200+ discourse features** including:
        
        - **Coreference Resolution** - Entity chains
        - **Entity Grid** - Role transitions
        - **Lexical Chains** - Semantic relations
        - **Constituency Parsing** - Syntax trees
        - **Entity Analysis** - Named entities
        - **POS Tagging** - Part of speech
        - **Readability** - Text complexity
        - **LFTK** - Linguistic features
        - **Coh-Metrix** - Cohesion indices
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Enter Your Text")
        text_input = st.text_area(
            "Paste your story or text here:",
            height=200,
            placeholder="Once upon a time, there was a little girl named Sarah. She lived in a small house near the forest with her mother and father. Every day, Sarah would walk through the forest to visit her grandmother who lived on the other side..."
        )
    
    with col2:
        st.subheader("📊 Text Statistics")
        if text_input:
            words = text_input.split()
            sentences = text_input.count('.') + text_input.count('!') + text_input.count('?')
            if sentences == 0:
                sentences = 1
            
            st.metric("Words", len(words))
            st.metric("Characters", len(text_input))
            st.metric("Sentences", sentences)
            st.metric("Avg Word Length", f"{len(text_input)/len(words):.1f}" if words else "0")
            st.metric("Avg Sentence Length", f"{len(words)/sentences:.1f}" if sentences else "0")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Text", 
            use_container_width=True, 
            type="primary",
            disabled=not health_status or not text_input
        )
    
    # Analysis
    if analyze_button and text_input and health_status:
        if len(text_input.split()) < 5:
            st.warning("⚠️ Text is very short. For meaningful analysis, please enter at least 50 words.")
        
        with st.spinner("🔄 Extracting 200+ discourse features... This may take 20-40 seconds."):
            start_time = time.time()
            
            try:
                # Call API
                response = requests.post(
                    API_URL, 
                    json={
                        "text": text_input,
                        "include_lftk": include_lftk,
                        "include_cohmetrix": include_cohmetrix
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    features = result.get('features', {})
                    categories = result.get('categories', {})
                    feature_count = result.get('feature_count', 0)
                    
                    elapsed_time = time.time() - start_time
                    
                    st.success(f"✅ Successfully extracted {feature_count} features in {elapsed_time:.1f} seconds!")
                    
                    # ====================================================
                    # SUMMARY METRICS
                    # ====================================================
                    st.header("📈 Analysis Summary")
                    
                    # Display category counts in columns
                    cols = st.columns(4)
                    category_items = list(categories.items())
                    for i in range(min(8, len(category_items))):
                        if i < 4:
                            with cols[i]:
                                cat, count = category_items[i]
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0">{cat.upper()}</h4>
                                    <h2 style="margin:0">{count}</h2>
                                    <small>features</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            if i == 4:
                                st.markdown("---")
                                cols2 = st.columns(4)
                            with cols2[i-4]:
                                cat, count = category_items[i]
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0">{cat.upper()}</h4>
                                    <h2 style="margin:0">{count}</h2>
                                    <small>features</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # ====================================================
                    # CATEGORIZED VISUALIZATIONS
                    # ====================================================
                    categorized = categorize_features(features)
                    
                    # Create tabs for different analysis categories
                    tabs = st.tabs([
                        "📖 Readability", 
                        "🏷️ Entity Analysis", 
                        "🔄 Discourse Structure",
                        "📊 POS & Grammar",
                        "📋 All Features"
                    ])
                    
                    # Tab 1: Readability
                    with tabs[0]:
                        st.subheader("📖 Readability & Text Complexity")
                        
                        # Readability metrics visualizations
                        plot_readability_metrics(features)
                        
                        # Aggregate statistics
                        agg_features = categorized.get('aggregates', {})
                        if agg_features:
                            st.subheader("Text Statistics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Create dataframe of aggregate features
                                agg_df = pd.DataFrame([
                                    {"Metric": k.replace('aggregates.', '').replace('_', ' ').title(), 
                                     "Value": f"{v:.2f}" if isinstance(v, float) else v}
                                    for k, v in agg_features.items()
                                ])
                                st.dataframe(agg_df, use_container_width=True, hide_index=True)
                            
                            with col2:
                                # Plot aggregate features
                                agg_plot_data = {k.replace('aggregates.', ''): v 
                                               for k, v in agg_features.items() 
                                               if isinstance(v, (int, float)) and v > 0}
                                if agg_plot_data:
                                    fig = px.line_polar(
                                        r=list(agg_plot_data.values()),
                                        theta=list(agg_plot_data.keys()),
                                        line_close=True,
                                        title="Text Statistics Radar"
                                    )
                                    fig.update_traces(fill='toself')
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tab 2: Entity Analysis
                    with tabs[1]:
                        st.subheader("🏷️ Entity Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Entity features
                            entity_features = categorized.get('ent', {})
                            if entity_features:
                                st.dataframe(
                                    pd.DataFrame([
                                        {"Feature": k.replace('ent.', '').replace('_', ' ').title(), 
                                         "Value": f"{v:.3f}"}
                                        for k, v in entity_features.items()
                                    ]),
                                    use_container_width=True,
                                    hide_index=True
                                )
                        
                        with col2:
                            # Entity grid transitions
                            eg_features = categorized.get('eg', {})
                            if eg_features:
                                plot_entity_grid_transitions(eg_features)
                        
                        # Entity composition plots
                        plot_entity_analysis(entity_features)
                    
                    # Tab 3: Discourse Structure
                    with tabs[2]:
                        st.subheader("🔄 Discourse Structure")
                        
                        # Get relevant features
                        coref_features = categorized.get('coref', {})
                        lex_features = categorized.get('lex', {})
                        const_features = categorized.get('const', {})
                        eg_features = categorized.get('eg', {})
                        
                        # Plot discourse structure
                        plot_discourse_structure(coref_features, lex_features, const_features)
                        
                        # Additional entity grid features
                        if eg_features:
                            with st.expander("📊 Entity Grid Details"):
                                eg_df = pd.DataFrame([
                                    {"Feature": k.replace('eg.', ''), "Value": f"{v:.4f}"}
                                    for k, v in eg_features.items()
                                    if not k.startswith('eg.trans')  # Filter out transitions
                                ])
                                st.dataframe(eg_df, use_container_width=True, hide_index=True)
                    
                    # Tab 4: POS & Grammar
                    with tabs[3]:
                        st.subheader("📊 Part-of-Speech Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # POS features
                            pos_features = categorized.get('pos', {})
                            if pos_features:
                                pos_df = pd.DataFrame([
                                    {"Feature": k.replace('pos.', '').replace('_', ' ').title(), 
                                     "Value": f"{v:.3f}"}
                                    for k, v in pos_features.items()
                                ])
                                st.dataframe(pos_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # POS radar chart
                            plot_pos_analysis(pos_features)
                        
                        # LFTK features if available
                        lftk_features = categorized.get('lftk', {})
                        if lftk_features:
                            with st.expander("📈 LFTK Features (Language Feature Toolkit)"):
                                lftk_df = pd.DataFrame([
                                    {"Feature": k.replace('lftk.', ''), "Value": f"{v:.4f}"}
                                    for k, v in lftk_features.items()
                                ])
                                st.dataframe(lftk_df, use_container_width=True, hide_index=True)
                    
                    # Tab 5: All Features
                    with tabs[4]:
                        st.subheader("📋 Complete Feature Set")
                        
                        # Create searchable dataframe
                        all_features_df = pd.DataFrame([
                            {
                                "Category": k.split('.')[0] if '.' in k else 'cohmetrix', 
                                "Feature": k, 
                                "Value": f"{v:.6f}" if isinstance(v, (int, float)) else str(v)
                            }
                            for k, v in features.items()
                        ])
                        
                        # Search and filter
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            search = st.text_input("🔍 Search features:", "", placeholder="Enter feature name...")
                        with col2:
                            category_filter = st.selectbox(
                                "Filter by category",
                                ["All"] + sorted(all_features_df['Category'].unique())
                            )
                        
                        # Apply filters
                        filtered_df = all_features_df
                        if search:
                            filtered_df = filtered_df[
                                filtered_df['Feature'].str.contains(search, case=False) |
                                filtered_df['Category'].str.contains(search, case=False)
                            ]
                        if category_filter != "All":
                            filtered_df = filtered_df[filtered_df['Category'] == category_filter]
                        
                        # Display with pagination
                        st.dataframe(
                            filtered_df,
                            use_container_width=True,
                            hide_index=True,
                            height=500
                        )
                        
                        # Download button
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"discourse_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    # Coh-Metrix section if available
                    cohmetrix_features = categorized.get('cohmetrix', {})
                    if cohmetrix_features and include_cohmetrix:
                        st.header("📊 Coh-Metrix Analysis")
                        
                        # Group Coh-Metrix features by prefix
                        coh_groups = {
                            'Descriptive': ['DESWC', 'DESSC', 'DESPL', 'DESSL'],
                            'Readability': ['RDFRE', 'RDFKGL', 'RDSMOG'],
                            'Lexical Diversity': ['LDTTRc', 'LDMTLD', 'LDVOCD'],
                            'Connectives': ['CNCAll', 'CNCCaus', 'CNCLogic'],
                            'Syntactic': ['SYNLE', 'SYNNP', 'SYNMEDpos']
                        }
                        
                        cols = st.columns(3)
                        for i, (group_name, prefixes) in enumerate(coh_groups.items()):
                            with cols[i % 3]:
                                group_features = {
                                    k: v for k, v in cohmetrix_features.items()
                                    if any(k.startswith(prefix) for prefix in prefixes)
                                }
                                if group_features:
                                    st.subheader(group_name)
                                    for k, v in group_features.items():
                                        st.metric(k.replace('cm.', ''), f"{v:.3f}")
                
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.json(response.text)
            
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The text might be too long or the backend is busy.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    elif analyze_button and not health_status:
        st.error("❌ Backend is not running. Please start the FastAPI server first.")
    
    elif analyze_button and not text_input:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Advanced Discourse Feature Analyzer | Extracts 200+ linguistic features</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()