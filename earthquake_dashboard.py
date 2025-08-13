import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Earthquake Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #ecf0f1;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        font-size: 1.8rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* Success metric */
    .success-metric {
        border-left-color: #27ae60;
    }
    
    .success-metric .metric-value {
        color: #27ae60;
    }
    
    /* Warning metric */
    .warning-metric {
        border-left-color: #f39c12;
    }
    
    .warning-metric .metric-value {
        color: #f39c12;
    }
    
    /* Danger metric */
    .danger-metric {
        border-left-color: #e74c3c;
    }
    
    .danger-metric .metric-value {
        color: #e74c3c;
    }
    
    /* Info metric */
    .info-metric {
        border-left-color: #3498db;
    }
    
    .info-metric .metric-value {
        color: #3498db;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        color: white;
        font-weight: 500;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9, #1f5f8b);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    
    /* Alert styling */
    .stAlert {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        color: #155724;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    /* Warning styling */
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        color: #856404;
        font-weight: 600;
    }
    
    /* Error styling */
    .stAlert[data-baseweb="notification"][data-testid="stAlert"] {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        color: #721c24;
        font-weight: 600;
    }
    
    /* Info styling */
    .stAlert[data-baseweb="notification"][data-testid="stAlert"] {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 2px solid #17a2b8;
        color: #0c5460;
        font-weight: 600;
    }
    
    /* Success styling */
    .stAlert[data-baseweb="notification"][data-testid="stAlert"] {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        color: #155724;
        font-weight: 600;
    }
    
    /* Ensure all text is visible */
    .stAlert * {
        color: inherit !important;
        font-weight: 600 !important;
    }
    
    /* Streamlit default alert overrides */
    .element-container .stAlert {
        background: white !important;
        border: 2px solid #3498db !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Warning messages */
    .element-container .stAlert[data-testid="stAlert"] {
        background: #fff3cd !important;
        border: 2px solid #ffc107 !important;
        color: #856404 !important;
        font-weight: 600 !important;
    }
    
    /* Error messages */
    .element-container .stAlert[data-testid="stAlert"] {
        background: #f8d7da !important;
        border: 2px solid #dc3545 !important;
        color: #721c24 !important;
        font-weight: 600 !important;
    }
    
    /* Info messages */
    .element-container .stAlert[data-testid="stAlert"] {
        background: #d1ecf1 !important;
        border: 2px solid #17a2b8 !important;
        color: #0c5460 !important;
        font-weight: 600 !important;
    }
    
    /* Success messages */
    .element-container .stAlert[data-testid="stAlert"] {
        background: #d4edda !important;
        border: 2px solid #28a745 !important;
        color: #155724 !important;
        font-weight: 600 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db, #2ecc71);
    }
    
    /* Text styling */
    .text-white {
        color: white !important;
    }
    
    .text-dark {
        color: #2c3e50 !important;
    }
    
    /* Spacing utilities */
    .mb-4 {
        margin-bottom: 2rem;
    }
    
    .mt-4 {
        margin-top: 2rem;
    }
    
    .p-4 {
        padding: 2rem;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the earthquake data"""
    try:
        df = pd.read_csv('database.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the data for analysis"""
    if df is None:
        return None
    
    df_processed = df.copy()
    
    # Convert Date and Time to datetime
    df_processed['DateTime'] = pd.to_datetime(df_processed['Date'] + ' ' + df_processed['Time'], errors='coerce')
    df_processed['Year'] = df_processed['DateTime'].dt.year
    df_processed['Month'] = df_processed['DateTime'].dt.month
    df_processed['Day'] = df_processed['DateTime'].dt.day
    
    # Handle missing values
    numeric_columns = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Fill missing values with median for numeric columns
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def create_metric_card(label, value, delta=None, metric_type="info"):
    """Create a styled metric card"""
    color_class = {
        "success": "success-metric",
        "warning": "warning-metric", 
        "danger": "danger-metric",
        "info": "info-metric"
    }.get(metric_type, "info-metric")
    
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Earthquake Analytics Dashboard</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
            Comprehensive Analysis & Machine Learning Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading earthquake data..."):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check if 'database.csv' exists in the current directory.")
        return
    
    # Preprocess data
    with st.spinner("🔄 Preprocessing data..."):
        df_processed = preprocess_data(df)
    
    if df_processed is None:
        st.error("❌ Failed to preprocess data.")
        return
    
    # Sidebar filters with modern styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50, #34495e); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">🔍 Data Filters</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Year filter
    years = sorted(df_processed['Year'].dropna().unique())
    
    # Add "Select All" option to the dropdown
    all_years_option = ["Select All"] + list(years)
    selected_years_raw = st.sidebar.multiselect(
        "📅 Select Years",
        options=all_years_option,
        default=years[:5] if len(years) > 5 else years,
        help="Choose specific years to analyze or select 'Select All' for all years"
    )
    
    # Handle "Select All" selection
    if "Select All" in selected_years_raw:
        selected_years = years
    else:
        selected_years = selected_years_raw
    
    # Magnitude filter
    mag_min = float(df_processed['Magnitude'].min())
    mag_max = float(df_processed['Magnitude'].max())
    mag_range = st.sidebar.slider(
        "🌊 Magnitude Range",
        min_value=mag_min,
        max_value=mag_max,
        value=(mag_min, mag_max),
        help="Filter earthquakes by magnitude"
    )
    min_mag, max_mag = mag_range
    
    # Depth filter
    depth_min = float(df_processed['Depth'].min())
    depth_max = float(df_processed['Depth'].max())
    depth_range = st.sidebar.slider(
        "⬇️ Depth Range (km)",
        min_value=depth_min,
        max_value=depth_max,
        value=(depth_min, depth_max),
        help="Filter earthquakes by depth"
    )
    min_depth, max_depth = depth_range
    
    # Type filter
    types = df_processed['Type'].unique()
    selected_types = st.sidebar.multiselect(
        "🎯 Event Type",
        options=types,
        default=types,
        help="Select specific event types"
    )
    
    # Apply filters
    filtered_df = df_processed[
        (df_processed['Year'].isin(selected_years)) &
        (df_processed['Magnitude'] >= min_mag) &
        (df_processed['Magnitude'] <= max_mag) &
        (df_processed['Depth'] >= min_depth) &
        (df_processed['Depth'] <= max_depth) &
        (df_processed['Type'].isin(selected_types))
    ]
    
    # Main content with modern tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Data Overview", "🔍 EDA", "📈 Visualizations", 
        "📊 Statistics", "🤖 ML Models", "🔍 Model Analysis", "💾 Export"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        # Key metrics in a modern grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card(
                "Total Records", 
                f"{len(filtered_df):,}", 
                metric_type="info"
            )
        
        with col2:
            create_metric_card(
                "Date Range", 
                f"{filtered_df['Year'].min()} - {filtered_df['Year'].max()}", 
                metric_type="info"
            )
        
        with col3:
            create_metric_card(
                "Avg Magnitude", 
                f"{filtered_df['Magnitude'].mean():.2f}", 
                metric_type="warning"
            )
        
        with col4:
            create_metric_card(
                "Avg Depth", 
                f"{filtered_df['Depth'].mean():.1f} km", 
                metric_type="info"
            )
        
        # Dataset overview with modern styling
        st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📄 Sample Data")
            st.dataframe(
                filtered_df.head(10),
                use_container_width=True,
                height=400
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 Dataset Info")
            st.write(f"**Shape:** {filtered_df.shape}")
            st.write(f"**Columns:** {len(filtered_df.columns)}")
            st.write(f"**Missing Values:** {filtered_df.isnull().sum().sum()}")
            st.write(f"**Unique Types:** {filtered_df['Type'].nunique()}")
            st.write(f"**Date Range:** {filtered_df['Year'].min()} - {filtered_df['Year'].max()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality analysis
        st.markdown('<div class="section-header">🔍 Data Quality Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📈 Data Types")
            dtype_df = pd.DataFrame({
                'Column': filtered_df.columns,
                'Data Type': filtered_df.dtypes,
                'Non-Null Count': filtered_df.count(),
                'Null Count': filtered_df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🔍 Missing Values")
            missing_data = filtered_df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(filtered_df)) * 100
            }).sort_values('Missing Count', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: EDA
    with tab2:
        st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Summary Statistics")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📈 Magnitude Distribution")
            fig_mag = px.histogram(
                filtered_df, x='Magnitude', nbins=50,
                title="Distribution of Earthquake Magnitudes",
                color_discrete_sequence=['#3498db']
            )
            fig_mag.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_mag, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📉 Depth Distribution")
            fig_depth = px.histogram(
                filtered_df, x='Depth', nbins=50,
                title="Distribution of Earthquake Depths",
                color_discrete_sequence=['#e74c3c']
            )
            fig_depth.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_depth, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Correlation analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔗 Correlation Analysis")
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Visualizations
    with tab3:
        st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
        
        # Time series analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("⏰ Time Series Analysis")
        
        yearly_counts = filtered_df.groupby('Year').size().reset_index(name='Count')
        fig_time = px.line(
            yearly_counts, x='Year', y='Count',
            title="Earthquakes Over Time",
            color_discrete_sequence=['#2ecc71']
        )
        fig_time.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Geographic and scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🎯 Magnitude vs Depth")
            fig_scatter = px.scatter(
                filtered_df, x='Magnitude', y='Depth',
                title="Magnitude vs Depth Relationship",
                color='Type',
                size='Magnitude',
                hover_data=['Latitude', 'Longitude']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🌍 Geographic Distribution")
            fig_map = px.scatter_mapbox(
                filtered_df,
                lat='Latitude',
                lon='Longitude',
                color='Magnitude',
                size='Magnitude',
                hover_data=['Date', 'Depth', 'Type'],
                title="Earthquake Locations",
                mapbox_style="open-street-map",
                zoom=1
            )
            fig_map.update_layout(height=400)
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Statistics
    with tab4:
        st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)
        
        # Statistical tests
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📈 Magnitude Statistics")
            mag_stats = filtered_df['Magnitude'].describe()
            st.write(f"**Mean:** {mag_stats['mean']:.3f}")
            st.write(f"**Median:** {mag_stats['50%']:.3f}")
            st.write(f"**Std Dev:** {mag_stats['std']:.3f}")
            st.write(f"**Skewness:** {filtered_df['Magnitude'].skew():.3f}")
            st.write(f"**Kurtosis:** {filtered_df['Magnitude'].kurtosis():.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📉 Depth Statistics")
            depth_stats = filtered_df['Depth'].describe()
            st.write(f"**Mean:** {depth_stats['mean']:.3f}")
            st.write(f"**Median:** {depth_stats['50%']:.3f}")
            st.write(f"**Std Dev:** {depth_stats['std']:.3f}")
            st.write(f"**Skewness:** {filtered_df['Depth'].skew():.3f}")
            st.write(f"**Kurtosis:** {filtered_df['Depth'].kurtosis():.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: ML Models
    with tab5:
        st.markdown('<div class="section-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)
        
        # Data Preparation Section
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔧 Data Preparation for Modeling")
        
        # Prepare data for modeling
        model_df = filtered_df.copy()
        
        # Create target variable (Magnitude category) with better binning
        # Check magnitude distribution first
        mag_stats = model_df['Magnitude'].describe()
        st.write("**📊 Magnitude Statistics:**")
        st.write(mag_stats)
        
        # Create more balanced categories based on data distribution
        mag_percentiles = model_df['Magnitude'].quantile([0.25, 0.5, 0.75, 0.9])
        
        # Use percentiles to create more balanced categories
        if mag_percentiles[0.9] < 7:
            # If 90th percentile is less than 7, use simpler categories
            bins = [0, mag_percentiles[0.25], mag_percentiles[0.5], mag_percentiles[0.75], 10]
            labels = ['Low', 'Medium', 'High', 'Very High']
        else:
            # Use standard magnitude categories
            bins = [0, 4, 5, 6, 7, 10]
            labels = ['Minor', 'Light', 'Moderate', 'Strong', 'Major']
        
        model_df['Magnitude_Category'] = pd.cut(
            model_df['Magnitude'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Check category distribution
        category_counts = model_df['Magnitude_Category'].value_counts()
        st.write("**📊 Magnitude Category Distribution:**")
        st.write(category_counts)
        
        # Remove categories with too few samples
        min_samples_per_category = 5
        valid_categories = category_counts[category_counts >= min_samples_per_category].index
        
        if len(valid_categories) < 2:
            st.markdown("""
            <div style="background: #fff3cd; border: 2px solid #ffc107; color: #856404; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                ⚠️ <strong>Not enough categories with sufficient samples. Trying with lower threshold...</strong>
            </div>
            """, unsafe_allow_html=True)
            min_samples_per_category = 2
            valid_categories = category_counts[category_counts >= min_samples_per_category].index
            
            if len(valid_categories) < 2:
                st.markdown("""
                <div style="background: #f8d7da; border: 2px solid #dc3545; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                    ❌ <strong>Still not enough categories. Please adjust the filters to include more diverse data.</strong>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div style="background: #d1ecf1; border: 2px solid #17a2b8; color: #0c5460; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                    💡 <strong>Try expanding the magnitude range or year range in the sidebar filters.</strong>
                </div>
                """, unsafe_allow_html=True)
                return
        
        model_df = model_df[model_df['Magnitude_Category'].isin(valid_categories)]
        
        if len(model_df) == 0:
            st.error("❌ No data available after filtering categories. Please adjust the filtering criteria.")
            return
        
        st.success(f"✅ After filtering: {len(model_df)} samples with {len(valid_categories)} categories")
        
        # Display target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Target Variable Distribution:**")
            target_dist = model_df['Magnitude_Category'].value_counts()
            st.write(target_dist)
            
            # Visualize target distribution
            fig_target = px.bar(
                x=target_dist.index,
                y=target_dist.values,
                title="Magnitude Category Distribution",
                color_discrete_sequence=['#3498db']
            )
            fig_target.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_target, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Feature Statistics:**")
            feature_columns = ['Latitude', 'Longitude', 'Depth']
            feature_stats = model_df[feature_columns].describe()
            st.dataframe(feature_stats, use_container_width=True)
        
        # Remove rows with missing values
        model_df_clean = model_df[feature_columns + ['Magnitude_Category']].dropna()
        
        if len(model_df_clean) == 0:
            st.error("❌ No data available for modeling after cleaning.")
            return
        
        st.success(f"✅ Cleaned dataset: {len(model_df_clean)} samples with {len(feature_columns)} features")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Training Section
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Model Training & Evaluation")
        
        # Encode target variable
        le = LabelEncoder()
        model_df_clean['Target_Encoded'] = le.fit_transform(model_df_clean['Magnitude_Category'])
        
        # Split data
        X = model_df_clean[feature_columns]
        y = model_df_clean['Target_Encoded']
        
        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        st.write("**📊 Class Distribution:**")
        st.write(class_counts)
        
        # Handle imbalanced classes
        min_class_count = class_counts.min()
        if min_class_count < 2:
            st.warning(f"⚠️ Some classes have very few samples. Minimum class count: {min_class_count}")
            st.info("ℹ️ Using random split instead of stratified split due to imbalanced classes.")
            
            # Use random split for imbalanced data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
        else:
            # Use stratified split for balanced data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Display split information
        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("Training Samples", f"{len(X_train):,}", metric_type="info")
        with col2:
            create_metric_card("Test Samples", f"{len(X_test):,}", metric_type="info")
        with col3:
            create_metric_card("Features", f"{len(feature_columns)}", metric_type="info")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check if we have enough data for training
        if len(X_train) < 10:
            st.markdown(f"""
            <div style="background: #f8d7da; border: 2px solid #dc3545; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                ❌ <strong>Insufficient training data. Need at least 10 samples, found {len(X_train)}.</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background: #d1ecf1; border: 2px solid #17a2b8; color: #0c5460; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                💡 <strong>Try adjusting the filters to include more data.</strong>
            </div>
            """, unsafe_allow_html=True)
            return
            
        if len(np.unique(y_train)) < 2:
            st.markdown(f"""
            <div style="background: #f8d7da; border: 2px solid #dc3545; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                ❌ <strong>Insufficient classes for training. Need at least 2 classes, found {len(np.unique(y_train))}.</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background: #d1ecf1; border: 2px solid #17a2b8; color: #0c5460; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                💡 <strong>Try adjusting the magnitude range or filters to get more diverse classes.</strong>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.success(f"✅ Ready to train models with {len(X_train)} training samples and {len(np.unique(y_train))} classes.")
        
        # Train multiple models with detailed parameters
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,  # Reduced for faster training
                    max_depth=8, 
                    min_samples_split=10,  # Increased for stability
                    min_samples_leaf=5,    # Increased for stability
                    random_state=42
                ),
                'description': 'Ensemble method using multiple decision trees'
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,  # Reduced for faster training
                    learning_rate=0.1,
                    max_depth=4,       # Reduced for stability
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ),
                'description': 'Sequential ensemble method with boosting'
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    C=1.0,
                    solver='lbfgs'
                ),
                'description': 'Linear model for classification'
            }
        }
        
        results = {}
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, model_info) in enumerate(models.items()):
            status_text.text(f"🔄 Training {name}...")
            
            model = model_info['model']
            
            # Check if we have enough classes and samples
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                st.warning(f"⚠️ {name}: Not enough classes for training. Need at least 2 classes, found {len(unique_classes)}.")
                continue
                
            # Check minimum samples per class
            class_counts = pd.Series(y_train).value_counts()
            min_samples = class_counts.min()
            if min_samples < 5:
                st.markdown(f"""
                <div style="background: #fff3cd; border: 2px solid #ffc107; color: #856404; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                    ⚠️ <strong>{name}: Some classes have very few samples (minimum: {min_samples}). Model may not perform well.</strong>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate detailed metrics
                accuracy = accuracy_score(y_test, y_pred)
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                # For multi-class, calculate macro averages with error handling
                try:
                    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                except:
                    precision = 0.0
                    
                try:
                    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                except:
                    recall = 0.0
                    
                try:
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                except:
                    f1 = 0.0
                
                # Calculate ROC AUC if possible
                try:
                    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    else:
                        roc_auc = None
                except:
                    roc_auc = None
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'predictions_proba': y_pred_proba,
                    'true_values': y_test,
                    'description': model_info['description']
                }
                
                st.success(f"✅ {name} trained successfully!")
                
            except Exception as e:
                st.markdown(f"""
                <div style="background: #f8d7da; border: 2px solid #dc3545; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                    ❌ <strong>{name} training failed: {str(e)}</strong>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background: #d1ecf1; border: 2px solid #17a2b8; color: #0c5460; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                    💡 <strong>This might be due to insufficient data or class imbalance. Try adjusting the filters.</strong>
                </div>
                """, unsafe_allow_html=True)
                continue
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.text("✅ Training completed!")
        
        # Check if any models were successfully trained
        if not results:
            st.markdown("""
            <div style="background: #f8d7da; border: 2px solid #dc3545; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600;">
                ❌ <strong>No models were successfully trained. Please try adjusting the data filters or use a different dataset.</strong>
            </div>
            """, unsafe_allow_html=True)
            return
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Comparison Section
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'] if result['roc_auc'] is not None else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig_acc = px.bar(
                comparison_df, 
                x='Model', 
                y='Accuracy',
                title="Model Accuracy Comparison",
                color_discrete_sequence=['#3498db']
            )
            fig_acc.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # F1-Score comparison
            fig_f1 = px.bar(
                comparison_df, 
                x='Model', 
                y='F1-Score',
                title="Model F1-Score Comparison",
                color_discrete_sequence=['#e74c3c']
            )
            fig_f1.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Find best model
        best_model = None
        best_score = 0
        
        for name, result in results.items():
            # Use F1-score as primary metric, fallback to accuracy
            score = result['f1_score'] if result['f1_score'] > 0 else result['accuracy']
            if score > best_score:
                best_score = score
                best_model = name
        
        st.success(f"🏆 Best Model: {best_model} (F1-Score: {results[best_model]['f1_score']:.3f})")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Model Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔍 Detailed Model Analysis")
        
        # Display detailed metrics for each model
        for name, result in results.items():
            with st.expander(f"📋 {name} - Detailed Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Model Description:** {result['description']}")
                    st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                    st.write(f"**Precision:** {result['precision']:.4f}")
                    st.write(f"**Recall:** {result['recall']:.4f}")
                    st.write(f"**F1-Score:** {result['f1_score']:.4f}")
                    if result['roc_auc'] is not None:
                        st.write(f"**ROC-AUC:** {result['roc_auc']:.4f}")
                
                with col2:
                    # Confusion matrix
                    cm = confusion_matrix(result['true_values'], result['predictions'])
                    fig_cm = px.imshow(
                        cm,
                        title=f"{name} - Confusion Matrix",
                        color_continuous_scale='Blues',
                        aspect='auto'
                    )
                    fig_cm.update_layout(height=300)
                    st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store best model for later use
        st.session_state['best_model'] = results[best_model]
        st.session_state['scaler'] = scaler
        st.session_state['label_encoder'] = le
        st.session_state['feature_columns'] = feature_columns
        st.session_state['all_results'] = results
    
    # Tab 6: Model Analysis
    with tab6:
        st.markdown('<div class="section-header">🔍 Model Analysis</div>', unsafe_allow_html=True)
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Please run the model training first.")
            return
        
        best_result = st.session_state['best_model']
        all_results = st.session_state['all_results']
        model_name = [k for k, v in all_results.items() if v == best_result][0]
        
        # Best Model Overview
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader(f"🏆 Best Model: {model_name}")
        
        # Key metrics for best model
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Accuracy", f"{best_result['accuracy']:.3f}", metric_type="success")
        with col2:
            create_metric_card("Precision", f"{best_result['precision']:.3f}", metric_type="info")
        with col3:
            create_metric_card("Recall", f"{best_result['recall']:.3f}", metric_type="warning")
        with col4:
            create_metric_card("F1-Score", f"{best_result['f1_score']:.3f}", metric_type="success")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Performance Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Detailed Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Model Performance Insights:**")
            st.write(f"• **Accuracy:** {best_result['accuracy']:.4f}")
            st.write(f"• **Precision:** {best_result['precision']:.4f}")
            st.write(f"• **Recall:** {best_result['recall']:.4f}")
            st.write(f"• **F1-Score:** {best_result['f1_score']:.4f}")
            if best_result['roc_auc'] is not None:
                st.write(f"• **ROC-AUC:** {best_result['roc_auc']:.4f}")
            st.write(f"• **Total Predictions:** {len(best_result['predictions']):,}")
            st.write(f"• **Correct Predictions:** {sum(best_result['predictions'] == best_result['true_values']):,}")
            st.write(f"• **Incorrect Predictions:** {sum(best_result['predictions'] != best_result['true_values']):,}")
            st.write(f"• **Error Rate:** {1 - best_result['accuracy']:.4f}")
        
        with col2:
            st.markdown("**🔧 Model Characteristics:**")
            st.write(f"• **Model Type:** {model_name}")
            st.write(f"• **Features Used:** {', '.join(st.session_state['feature_columns'])}")
            st.write(f"• **Target Classes:** {len(st.session_state['label_encoder'].classes_)}")
            st.write(f"• **Training Samples:** {len(X_train):,}")
            st.write(f"• **Test Samples:** {len(X_test):,}")
            st.write(f"• **Feature Scaling:** StandardScaler")
            st.write(f"• **Cross-Validation:** Stratified Split (80/20)")
            st.write(f"• **Random State:** 42")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔍 Confusion Matrix Analysis")
        
        # Create confusion matrix
        cm = confusion_matrix(best_result['true_values'], best_result['predictions'])
        class_names = st.session_state['label_encoder'].classes_
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visual confusion matrix
            fig_cm = px.imshow(
                cm,
                x=class_names,
                y=class_names,
                title=f"{model_name} - Confusion Matrix",
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Confusion matrix statistics
            st.markdown("**📈 Confusion Matrix Analysis:**")
            
            # Calculate per-class metrics
            for i, class_name in enumerate(class_names):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                st.write(f"**{class_name}:**")
                st.write(f"  • Precision: {precision:.3f}")
                st.write(f"  • Recall: {recall:.3f}")
                st.write(f"  • F1-Score: {f1:.3f}")
                st.write(f"  • Support: {cm[i, :].sum()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Importance Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Feature Importance Analysis")
        
        if hasattr(best_result['model'], 'feature_importances_'):
            # Tree-based models
            importance_df = pd.DataFrame({
                'Feature': st.session_state['feature_columns'],
                'Importance': best_result['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance bar chart
                fig_importance = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title=f"{model_name} - Feature Importance",
                    color_discrete_sequence=['#3498db']
                )
                fig_importance.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                # Feature importance table
                st.markdown("**📊 Feature Importance Rankings:**")
                st.dataframe(importance_df, use_container_width=True)
                
                # Feature importance insights
                st.markdown("**💡 Feature Importance Insights:**")
                most_important = importance_df.iloc[0]
                st.write(f"• **Most Important Feature:** {most_important['Feature']} ({most_important['Importance']:.3f})")
                st.write(f"• **Least Important Feature:** {importance_df.iloc[-1]['Feature']} ({importance_df.iloc[-1]['Importance']:.3f})")
                st.write(f"• **Importance Range:** {importance_df['Importance'].max():.3f} - {importance_df['Importance'].min():.3f}")
        
        elif hasattr(best_result['model'], 'coef_'):
            # Linear models
            coef_df = pd.DataFrame({
                'Feature': st.session_state['feature_columns'],
                'Coefficient': best_result['model'].coef_[0] if len(best_result['model'].coef_.shape) > 1 else best_result['model'].coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Coefficient bar chart
                fig_coef = px.bar(
                    coef_df,
                    x='Feature',
                    y='Coefficient',
                    title=f"{model_name} - Feature Coefficients",
                    color_discrete_sequence=['#e74c3c']
                )
                fig_coef.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_coef, use_container_width=True)
            
            with col2:
                # Coefficient table
                st.markdown("**📊 Feature Coefficient Rankings:**")
                st.dataframe(coef_df, use_container_width=True)
                
                # Coefficient insights
                st.markdown("**💡 Coefficient Insights:**")
                most_positive = coef_df.loc[coef_df['Coefficient'].idxmax()]
                most_negative = coef_df.loc[coef_df['Coefficient'].idxmin()]
                st.write(f"• **Most Positive Impact:** {most_positive['Feature']} ({most_positive['Coefficient']:.3f})")
                st.write(f"• **Most Negative Impact:** {most_negative['Feature']} ({most_negative['Coefficient']:.3f})")
                st.write(f"• **Coefficient Range:** {coef_df['Coefficient'].max():.3f} - {coef_df['Coefficient'].min():.3f}")
        
        else:
            st.info("ℹ️ Feature importance not available for this model type.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔮 Prediction Analysis")
        
        # Create prediction analysis dataframe
        pred_analysis = pd.DataFrame({
            'True_Label': best_result['true_values'],
            'Predicted_Label': best_result['predictions'],
            'Correct': best_result['predictions'] == best_result['true_values']
        })
        
        # Add class names
        pred_analysis['True_Class'] = st.session_state['label_encoder'].inverse_transform(pred_analysis['True_Label'])
        pred_analysis['Predicted_Class'] = st.session_state['label_encoder'].inverse_transform(pred_analysis['Predicted_Label'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy by class
            class_accuracy = pred_analysis.groupby('True_Class')['Correct'].agg(['mean', 'count'])
            class_accuracy.columns = ['Accuracy', 'Count']
            
            fig_class_acc = px.bar(
                class_accuracy,
                y='Accuracy',
                title="Accuracy by Magnitude Category",
                color_discrete_sequence=['#e74c3c' if x < 0.5 else '#27ae60' for x in class_accuracy['Accuracy']]
            )
            fig_class_acc.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_class_acc, use_container_width=True)
        
        with col2:
            # Error analysis
            errors = pred_analysis[~pred_analysis['Correct']]
            
            if len(errors) > 0:
                st.markdown("**❌ Error Analysis:**")
                st.write(f"• **Total Errors:** {len(errors)}")
                st.write(f"• **Error Rate:** {len(errors)/len(pred_analysis):.3f}")
                
                # Most common errors
                error_counts = errors.groupby(['True_Class', 'Predicted_Class']).size().reset_index(name='Count')
                error_counts = error_counts.sort_values('Count', ascending=False)
                
                st.write("**Most Common Prediction Errors:**")
                st.dataframe(error_counts.head(5), use_container_width=True)
            else:
                st.success("🎉 Perfect predictions! No errors found.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Confidence Analysis
        if best_result['predictions_proba'] is not None:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🎲 Prediction Confidence Analysis")
            
            # Calculate confidence scores
            confidence = np.max(best_result['predictions_proba'], axis=1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence distribution
                fig_confidence = px.histogram(
                    x=confidence,
                    title="Distribution of Prediction Confidence",
                    nbins=30,
                    color_discrete_sequence=['#9b59b6']
                )
                fig_confidence.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            with col2:
                # Confidence statistics
                st.markdown("**📊 Confidence Statistics:**")
                st.write(f"• **Average Confidence:** {confidence.mean():.3f}")
                st.write(f"• **Confidence Std Dev:** {confidence.std():.3f}")
                st.write(f"• **Min Confidence:** {confidence.min():.3f}")
                st.write(f"• **Max Confidence:** {confidence.max():.3f}")
                st.write(f"• **High Confidence (>0.8):** {(confidence > 0.8).sum()}")
                st.write(f"• **Low Confidence (<0.5):** {(confidence < 0.5).sum()}")
                
                # Confidence vs accuracy
                high_conf_mask = confidence > 0.8
                if high_conf_mask.sum() > 0:
                    high_conf_accuracy = pred_analysis.loc[high_conf_mask, 'Correct'].mean()
                    st.write(f"• **High Confidence Accuracy:** {high_conf_accuracy:.3f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Comparison Summary
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Model Comparison Summary")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in all_results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'] if result['roc_auc'] is not None else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Highlight best model
        comparison_df['Best'] = comparison_df['Model'] == model_name
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model comparison table
            st.markdown("**📋 All Models Performance:**")
            st.dataframe(comparison_df, use_container_width=True)
        
        with col2:
            # Model ranking
            st.markdown("**🏆 Model Rankings:**")
            
            # Rank by different metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for metric in metrics:
                best_model_for_metric = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
                best_value = comparison_df[metric].max()
                st.write(f"• **Best {metric}:** {best_model_for_metric} ({best_value:.3f})")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 7: Export
    with tab7:
        st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Cleaned Dataset Export**")
            
            if st.button("📥 Export Cleaned CSV", key="export_csv"):
                cleaned_df = filtered_df.copy()
                cleaned_df = cleaned_df.dropna()
                cleaned_df.to_csv("cleaned_earthquake_data.csv", index=False)
                st.success("✅ Cleaned data exported as 'cleaned_earthquake_data.csv'")
                
                with open("cleaned_earthquake_data.csv", "r") as f:
                    st.download_button(
                        label="📥 Download Cleaned CSV",
                        data=f.read(),
                        file_name="cleaned_earthquake_data.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.markdown("**🤖 Model Export**")
            
            if 'best_model' in st.session_state:
                import pickle
                
                if st.button("📥 Export Best Model", key="export_model"):
                    model_package = {
                        'model': st.session_state['best_model']['model'],
                        'scaler': st.session_state['scaler'],
                        'label_encoder': st.session_state['label_encoder'],
                        'feature_columns': st.session_state['feature_columns'],
                        'accuracy': st.session_state['best_model']['accuracy']
                    }
                    
                    with open("best_earthquake_model.pkl", "wb") as f:
                        pickle.dump(model_package, f)
                    
                    st.success("✅ Best model exported as 'best_earthquake_model.pkl'")
                    
                    with open("best_earthquake_model.pkl", "rb") as f:
                        st.download_button(
                            label="📥 Download Model",
                            data=f.read(),
                            file_name="best_earthquake_model.pkl",
                            mime="application/octet-stream"
                        )
            else:
                st.warning("⚠️ No trained model available. Please run model training first.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 