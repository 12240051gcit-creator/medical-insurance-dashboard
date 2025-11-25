import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# ==================== PAGE CONFIG & CSS ====================
st.set_page_config(page_title="Medical Insurance Dashboard", layout="wide")
st.markdown("""
<style>
    .main > div {padding-top: 0rem;}
    .stat-card {
        background: white; padding: 1.8rem; border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12); text-align: center;
        transition: all 0.3s; height: 100%;
    }
    .stat-card:hover {transform: translateY(-8px); box-shadow: 0 20px 40px rgba(0,0,0,0.18);}
    .stat-card h3 {color: #555; font-size: 1.1rem; margin: 0 0 10px 0;}
    .stat-card .big-num {font-size: 2.8rem; font-weight: 800; color: #D32F2F; margin: 8px 0;}
   
    .wallet-card {
        background: #D32F2F;
        color: white; padding: 1.6rem; border-radius: 16px;
        box-shadow: 0 10px 30px rgba(211,47,47,0.4); text-align: center;
    }
    .small-wallet {padding: 1.2rem; margin-top: 12px;}
    .small-wallet:nth-child(2) {background: #C2185B;}
    .small-wallet:nth-child(3) {background: #B71C1C;}
   
    .activity-item {
        padding: 14px 0; border-bottom: 1px solid #f0f0f0; display: flex; gap: 12px; align-items: center;
    }
    .activity-item:last-child {border-bottom: none;}
    
    .stButton>button {
        background: #D32F2F !important; color: white !important;
        border-radius: 12px !important; padding: 12px 32px !important;
        font-weight: bold !important; border: none !important; width: 100%;
    }
    
    .input-section {
        background: #f8f9fa; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
    }
    
    .result-card {
        background: #D32F2F;
        color: white; padding: 2rem; border-radius: 16px; text-align: center;
        box-shadow: 0 10px 30px rgba(211,47,47,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA & MODEL ====================
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

try:
    df = load_data()
    
    # Encode for model
    X = pd.get_dummies(df.drop("charges", axis=1), drop_first=True)
    y = df["charges"]
    model = LinearRegression().fit(X, y)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ==================== SIMPLIFIED SIDEBAR ====================
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#D32F2F;'>Insurance Cost Analysis</h2>", unsafe_allow_html=True)
    st.markdown("---")
   
    page = st.radio("Navigate", [
        "Overview",
        "Data Analysis",
        "Visualizations", 
        "Insights",
        "Premium Calculator"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.caption("Group 2 • GCIT")

# ==================== HEADER ====================
st.markdown("<h1 style='text-align:center; color:#D32F2F; margin:0;'>Medical Insurance</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#D32F2F; opacity:0.9; margin-top:8px;'>Cost Analysis</h2>", unsafe_allow_html=True)
st.markdown("---")

# ==================== TOP METRICS ====================
col1, col2, col3, col4 = st.columns(4)
avg_charge = df.charges.mean()
smoker_multiplier = df[df.smoker=='yes'].charges.mean() / df[df.smoker=='no'].charges.mean()
age_effect = model.coef_[0] # From linear model
obese_premium = (df[df.bmi >= 30].charges.mean() / df[(df.bmi >= 18.5) & (df.bmi < 25)].charges.mean() - 1) * 100

with col1:
    st.markdown(f'''
    <div class="stat-card">
        <h3>Average Charge</h3>
        <div class="big-num">${avg_charge:,.0f}</div>
        <small>(n=1,338)</small>
    </div>
    ''', unsafe_allow_html=True)
with col2:
    st.markdown(f'''
    <div class="stat-card">
        <h3>Smokers Pay</h3>
        <div class="big-num">{smoker_multiplier:.1f}×</div>
        <small>More than non-smokers</small>
    </div>
    ''', unsafe_allow_html=True)
with col3:
    st.markdown(f'''
    <div class="stat-card">
        <h3>Age Effect</h3>
        <div class="big-num">+${age_effect:.0f}/yr</div>
        <small>Cost increase per year</small>
    </div>
    ''', unsafe_allow_html=True)
with col4:
    st.markdown(f'''
    <div class="stat-card">
        <h3>Obesity Premium</h3>
        <div class="big-num">+{obese_premium:.0f}%</div>
        <small>For BMI ≥30</small>
    </div>
    ''', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==================== MAIN LAYOUT ====================
left_col, right_col = st.columns([2, 1])

with left_col:
    if page == "Overview":
        st.subheader("Project Overview")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://img.icons8.com/color/200/health-checkup.png", width=150)
        with col2:
            st.markdown("Key Objectives:")
            st.markdown("• Explore factors influencing medical insurance costs")
            st.markdown("• Identify patterns and relationships between variables")
            st.markdown("• Provide data-driven insights for pricing strategies")
            st.markdown("• Develop predictive model for premium estimation")
        
        # Research Questions
        st.subheader("Research Questions & Hypotheses")
       
        with st.expander("R1 & H1: Age-Smoking Relationship"):
            st.markdown("""
            R1: How does the relationship between age and medical insurance costs differ between smokers and non-smokers?
           
            H1: The relationship is non-linear and accelerating, significantly steeper for smokers.
            """)
       
        with st.expander("R2 & H2: Regional Cost Premium"):
            st.markdown("""
            R2: Does the Southeast region maintain a statistically significant cost premium over the Southwest?
           
            H2: Yes — disparity persists even among low-risk individuals.
            """)
       
        with st.expander("R3 & H3: Smoking Dominance"):
            st.markdown("""
            R3: Is the smoking cost premium the dominant factor in medical expenses?
           
            H3: Yes — largest single cost difference in the dataset.
            """)

    elif page == "Data Analysis":
        st.subheader("Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Dataset Features")
            st.markdown("• Categorical: sex, smoker, region")
            st.markdown("• Numerical: age, bmi, children, charges")
            st.success("No missing values\nClean & valid data")
        with col2:
            st.markdown("Key Findings")
            st.markdown("• Smoking: Dominant cost driver")
            st.markdown("• Age: Steady cost increase")
            st.markdown("• BMI: Gradual cost escalation")
            st.markdown("• Region: Southeast highest costs")
        
        # Show data sample
        st.markdown("Data Sample:")
        st.table(df.head(10).reset_index(drop=True))
        
        # Data summary
        st.markdown("Data Summary:")
        st.write(f"Total records: {len(df)}")
        st.write(f"Columns: {', '.join(df.columns.tolist())}")

    elif page == "Visualizations":
        st.subheader("Interactive Data Visualizations")
       
        # Filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            x_var = st.selectbox("X-Axis", ["age", "bmi", "children"], index=0)
        with col_filter2:
            y_var = st.selectbox("Y-Axis", ["charges", "bmi", "age"], index=0)
        with col_filter3:
            color_by = st.selectbox("Color By", ["smoker", "sex", "region"])
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df[x_var], df[y_var], 
                            c=df[color_by].astype('category').cat.codes,
                            cmap="viridis", s=60, alpha=0.7, edgecolors="white")
        ax.set_xlabel(x_var.capitalize())
        ax.set_ylabel(y_var.capitalize())
        ax.set_title(f"{x_var.capitalize()} vs {y_var.capitalize()} (colored by {color_by})", fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        # Create legend
        handles, labels = scatter.legend_elements()
        ax.legend(handles, df[color_by].unique(), title=color_by.capitalize(), bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)
        
        # Smoking Impact Filterable View
        st.markdown("#### Smoking Impact — Filterable View")
        age_range = st.slider("Age Range", 18, 64, (18, 64), key="age_slider")
        bmi_range = st.slider("BMI Range", 15.0, 50.0, (18.5, 40.0), key="bmi_slider")
        
        filtered = df[(df.age.between(age_range[0], age_range[1])) & 
                     (df.bmi.between(bmi_range[0], bmi_range[1]))]
       
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=filtered, x="smoker", y="charges", palette=["#a8dadc", "#D32F2F"], ax=ax2)
        sns.stripplot(data=filtered, x="smoker", y="charges", color="black", alpha=0.4, jitter=True, ax=ax2)
        ax2.set_title(f"Charges Distribution (Age {age_range[0]}-{age_range[1]}, BMI {bmi_range[0]}-{bmi_range[1]})")
        ax2.set_xticklabels(['Non-Smoker', 'Smoker'])
        st.pyplot(fig2)

    elif page == "Insights":
        st.subheader("Insights and Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Major Insights:")
            st.markdown("• Smoking Premium: Smokers pay 3.8× more")
            st.markdown("• Age Progression: Costs increase $257 per year")
            st.markdown("• BMI Penalty: Obese individuals pay 40% more")
            st.markdown("• Regional Variation: Southeast costs 15% higher")
            
        with col2:
            st.markdown("Recommendations:")
            st.markdown("• Smoking-focused pricing strategy")
            st.markdown("• Age-based pricing adjustments")
            st.markdown("• Wellness programs for BMI management")
            st.markdown("• Regional pricing optimization")
        
        # Correlation matrix
        st.markdown("#### Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax_corr)
        ax_corr.set_title('Correlation Matrix of Numerical Features')
        st.pyplot(fig_corr)

    elif page == "Premium Calculator":
        st.subheader("Premium Calculator")
        st.markdown("Enter your details below to get an instant insurance premium estimate")
        
        # Personal Information Section
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            age = st.slider("Age", 18, 100, 35, help="Your current age")
            bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1, 
                                help="Body Mass Index (Normal: 18.5-24.9)")
            children = st.selectbox("Number of Children", [0,1,2,3,4,5],
                                  help="Dependents covered by insurance")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            sex = st.radio("Gender", ["male", "female"], horizontal=True)
            smoker = st.radio("Smoker", ["no", "yes"], horizontal=True,
                            help="Tobacco usage significantly affects premiums")
            region = st.selectbox("Region", df['region'].unique(),
                                help="Your geographic location")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate Button
        col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
        with col_btn2:
            calculate_clicked = st.button("Calculate My Premium", use_container_width=True)
        
        if calculate_clicked:
            # Prepare input data
            input_df = pd.DataFrame({
                'age': [age], 'bmi': [bmi], 'children': [children],
                'sex': [sex], 'smoker': [smoker], 'region': [region]
            })
            input_encoded = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
            predicted = model.predict(input_encoded)[0]
            
            # Display Results
            st.markdown("---")
            st.markdown(f'''
            <div class="result-card">
                <h2 style="margin:0; font-size:1.8rem;">Estimated Annual Premium</h2>
                <h1 style="font-size:3.5rem; margin:1rem 0; font-weight:bold;">${predicted:,.0f}</h1>
                <p style="margin:0; font-size:1.1rem;">USD per year • ${predicted/12:,.0f}/month</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Additional Insights
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                if smoker == "yes":
                    non_smoker_cost = predicted / smoker_multiplier
                    savings = predicted - non_smoker_cost
                    st.error(f"""
                    Smoking Impact  
                    Quit smoking to save approximately:  
                    ${savings:,.0f} per year
                    """)
                else:
                    st.success("""
                    Non-Smoker Benefit  
                    You're saving significantly by not smoking!
                    """)
            
            with col_insight2:
                bmi_category = "Normal" if 18.5 <= bmi <= 24.9 else "Overweight" if 25 <= bmi <= 29.9 else "Obese"
                if bmi_category != "Normal":
                    st.warning(f"""
                    BMI Alert  
                    Your BMI ({bmi}) is in {bmi_category} range  
                    Consider wellness programs for better rates
                    """)
                else:
                    st.success("""
                    Healthy BMI  
                    Your BMI is in the healthy range!
                    """)
            
            # Comparison with averages
            st.markdown("#### How You Compare")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                diff_vs_avg = ((predicted - avg_charge) / avg_charge) * 100
                st.metric(
                    "vs Average Premium", 
                    f"${predicted:,.0f}", 
                    f"{diff_vs_avg:+.1f}%"
                )
            
            with comp_col2:
                similar_age = df[df.age.between(age-2, age+2)].charges.mean()
                diff_vs_age = ((predicted - similar_age) / similar_age) * 100
                st.metric(
                    "vs Similar Age", 
                    f"${predicted:,.0f}", 
                    f"{diff_vs_age:+.1f}%"
                )
            
            with comp_col3:
                similar_profile = df[(df.smoker == smoker) & (df.age.between(age-5, age+5))].charges.mean()
                diff_vs_profile = ((predicted - similar_profile) / similar_profile) * 100
                st.metric(
                    "vs Similar Profile", 
                    f"${predicted:,.0f}", 
                    f"{diff_vs_profile:+.1f}%"
                )

with right_col:
    # Always show wallet cards and recommendations in right column
    st.markdown('''
    <div class="wallet-card">
        <h4>Smoking Premium</h4>
        <h2>3.8×</h2>
    </div>
    <div class="wallet-card small-wallet">
        <h4>Age Increase</h4>
        <h3>$257/year</h3>
    </div>
    <div class="wallet-card small-wallet">
        <h4>BMI Penalty</h4>
        <h3>+40%</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Key Recommendations")
    recs = [
        ("Smoking Strategy", "300% premium for smokers"),
        ("Age Pricing", "8% annual increase"),
        ("Wellness Programs", "15% discount for healthy BMI"),
        ("Regional Opt.", "15-20% adj. for Southeast")
    ]
    for title, desc in recs:
        st.markdown(f'''
        <div class="activity-item">
            <div>
                <div style="font-weight:600;">{title}</div>
                <small style="color:#888;">{desc}</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>© 2025 Group 2 • Gyalpozhing College of Information Technology</p>", unsafe_allow_html=True)