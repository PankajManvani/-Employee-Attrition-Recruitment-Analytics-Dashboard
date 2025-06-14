"""
Employee Position Fill Time Dashboard
A Streamlit application for visualizing how long it takes to fill positions after employee attrition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Position Fill Time Analysis",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard title and description
st.title("Position Fill Time Analysis Dashboard")
st.write("""
This dashboard analyzes how long it takes to fill positions after employees leave the company.
It provides insights into fill times across departments, roles, and seniority levels to help optimize
recruitment strategies and reduce vacancy periods.
""")

# Sidebar
st.sidebar.header("Dashboard Controls")

# Helper function to process the dataset with automatic column mapping
def process_dataset(data_df):
    """
    Process a dataset to make it compatible with the dashboard.
    Attempts to map columns from various HR data formats to the required format.
    """
    required_columns = [
        'vacancy_date', 'filled_date', 'days_to_fill', 'department', 
        'job_role', 'seniority_level', 'reason_for_leaving', 'candidate_source'
    ]
    
    # Check if all required columns already exist
    if all(col in data_df.columns for col in required_columns):
        # Just convert date columns
        try:
            data_df['vacancy_date'] = pd.to_datetime(data_df['vacancy_date'])
            data_df['filled_date'] = pd.to_datetime(data_df['filled_date'])
            return data_df, True, "Dataset loaded successfully!"
        except Exception as e:
            return None, False, f"Error converting date columns: {str(e)}"
    
    # Try to automatically map columns based on common HR dataset formats
    column_mappings = {
        # Common alternatives for each required column
        'vacancy_date': ['vacancy_date', 'attrition_date', 'termination_date', 'separation_date', 'leave_date'],
        'filled_date': ['filled_date', 'hire_date', 'replacement_date', 'start_date'],
        'days_to_fill': ['days_to_fill', 'time_to_fill', 'vacancy_duration', 'hiring_time'],
        'department': ['department', 'Department', 'dept', 'business_unit', 'division'],
        'job_role': ['job_role', 'JobRole', 'role', 'position', 'title', 'job_title'],
        'seniority_level': ['seniority_level', 'JobLevel', 'level', 'grade', 'seniority'],
        'reason_for_leaving': ['reason_for_leaving', 'attrition_reason', 'termination_reason', 'separation_reason'],
        'candidate_source': ['candidate_source', 'source', 'recruitment_source', 'hiring_source']
    }
    
    # Try to map existing columns to required ones
    mapped_data = {}
    found_mappings = {}
    missing_columns = []
    
    for req_col, alternatives in column_mappings.items():
        mapped = False
        for alt in alternatives:
            if alt in data_df.columns:
                mapped_data[req_col] = data_df[alt]
                found_mappings[req_col] = alt
                mapped = True
                break
        
        if not mapped:
            missing_columns.append(req_col)
    
    # Special handling: if we're missing date columns but have attrition data
    # Try to generate vacancy dates from available data
    if ('vacancy_date' in missing_columns or 'filled_date' in missing_columns) and 'Attrition' in data_df.columns:
        # Generate synthetic dates if working with an attrition dataset
        if 'vacancy_date' in missing_columns:
            # Use today as a base date and go back randomly up to a year
            base_date = datetime.now() - pd.Timedelta(days=365)
            random_days = pd.Series(np.random.randint(0, 365, size=len(data_df)))
            mapped_data['vacancy_date'] = base_date + pd.to_timedelta(random_days, unit='d')
            missing_columns.remove('vacancy_date')
            found_mappings['vacancy_date'] = 'generated from attrition data'
        
        if 'filled_date' in missing_columns and 'vacancy_date' in mapped_data:
            # Generate filled dates by adding days_to_fill to vacancy_date
            if 'days_to_fill' in missing_columns:
                # Make realistic fill times based on job level or randomly if no job level
                if 'JobLevel' in data_df.columns:
                    # Higher levels take longer to fill
                    days_map = {1: 20, 2: 30, 3: 45, 4: 60, 5: 90}
                    mapped_data['days_to_fill'] = data_df['JobLevel'].map(days_map).fillna(30)
                else:
                    mapped_data['days_to_fill'] = pd.Series(np.random.randint(15, 90, size=len(data_df)))
                
                missing_columns.remove('days_to_fill')
                found_mappings['days_to_fill'] = 'generated based on attributes'
            
            mapped_data['filled_date'] = mapped_data['vacancy_date'] + pd.to_timedelta(mapped_data['days_to_fill'], unit='d')
            missing_columns.remove('filled_date')
            found_mappings['filled_date'] = 'calculated from vacancy_date and days_to_fill'
    
    # If we still have missing columns but have 'Attrition' and 'Department'
    if missing_columns and 'Attrition' in data_df.columns:
        # Keep only attrition=Yes records if we're working with attrition data
        attrition_mask = data_df['Attrition'] == 'Yes'
        for col in data_df.columns:
            if col in mapped_data:
                mapped_data[col] = mapped_data[col][attrition_mask].reset_index(drop=True)
        
        # Filter the original dataframe
        filtered_df = data_df[attrition_mask].reset_index(drop=True)
        
        # Generate candidate source if missing
        if 'candidate_source' in missing_columns:
            sources = ['Internal Referral', 'LinkedIn', 'External Job Board', 
                       'Company Website', 'Recruiting Agency', 'Career Fair',
                       'Headhunter', 'Direct Application']
            mapped_data['candidate_source'] = pd.Series(np.random.choice(sources, size=len(filtered_df)))
            missing_columns.remove('candidate_source')
            found_mappings['candidate_source'] = 'generated with realistic values'
        
        # Generate reason for leaving based on available factors
        if 'reason_for_leaving' in missing_columns:
            def generate_reason(row):
                if 'JobSatisfaction' in row and row['JobSatisfaction'] <= 2:
                    return 'Dissatisfaction'
                elif 'WorkLifeBalance' in row and row['WorkLifeBalance'] <= 2:
                    return 'Work-Life Balance'
                elif 'EnvironmentSatisfaction' in row and row['EnvironmentSatisfaction'] <= 2:
                    return 'Work Environment'
                elif 'MonthlyIncome' in row and row['MonthlyIncome'] < 5000:
                    return 'Better Opportunity'
                elif 'YearsAtCompany' in row and row['YearsAtCompany'] > 15:
                    return 'Retirement'
                else:
                    return pd.Series(['Personal Reasons', 'Relocation', 'Career Change', 
                                     'Resignation']).sample(1).iloc[0]
            
            mapped_data['reason_for_leaving'] = filtered_df.apply(generate_reason, axis=1)
            missing_columns.remove('reason_for_leaving')
            found_mappings['reason_for_leaving'] = 'derived from employee attributes'
    
    # If we still have missing critical columns
    if 'vacancy_date' in missing_columns or 'filled_date' in missing_columns or 'days_to_fill' in missing_columns:
        return None, False, f"Could not map or generate the critical columns: {', '.join(missing_columns)}"
    
    # Create new dataframe with mapped columns
    result_df = pd.DataFrame(mapped_data)
    
    # Generate any remaining missing columns with defaults
    if 'department' in missing_columns:
        result_df['department'] = 'Not Specified'
        found_mappings['department'] = 'default value'
    
    if 'job_role' in missing_columns:
        result_df['job_role'] = 'Not Specified'
        found_mappings['job_role'] = 'default value'
    
    if 'seniority_level' in missing_columns:
        result_df['seniority_level'] = 'Mid-level'
        found_mappings['seniority_level'] = 'default value'
    
    # Create success message with mapping info
    mapping_info = "\n".join([f"• {req}: {src}" for req, src in found_mappings.items()])
    success_msg = f"Dataset processed successfully!\n\nColumn mappings used:\n{mapping_info}"
    
    return result_df, True, success_msg

# File upload/selection section
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio(
    "Select data source:",
    ["Use default dataset", "Upload file"]
)

# Initialize dataframe
df = None

if data_source == "Use default dataset":
    # Load default dataset
    default_data_path = "data/position_fill_times.csv"
    
    if os.path.exists(default_data_path):
        try:
            raw_df = pd.read_csv(default_data_path)
            df, success, message = process_dataset(raw_df)
            
            if success:
                st.sidebar.success("Default dataset loaded successfully!")
            else:
                st.sidebar.error(message)
                df = None
        except Exception as e:
            st.sidebar.error(f"Error loading default dataset: {str(e)}")
            df = None
    else:
        st.sidebar.error("Default dataset not found!")
else:
    # File upload option
    st.sidebar.info("Please upload your CSV file with HR data")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            
            # Display file info
            st.sidebar.write(f"File uploaded: {uploaded_file.name}")
            st.sidebar.write(f"Columns found: {len(raw_df.columns)}")
            st.sidebar.write(f"Records found: {len(raw_df)}")
            
            # Process the uploaded dataset
            df, success, message = process_dataset(raw_df)
            
            if success:
                st.sidebar.success(message)
                
                # Show a sample of the processed data
                with st.sidebar.expander("Preview processed data"):
                    st.dataframe(df.head(3) if df is not None else None)
            else:
                st.sidebar.error(message)
                # Offer suggestions for required format
                st.sidebar.info("""
                Required columns:
                - vacancy_date: When position became vacant
                - filled_date: When position was filled
                - days_to_fill: Days between vacancy and filling
                - department: Department name
                - job_role: Position title 
                - seniority_level: Junior/Mid/Senior/Executive
                - reason_for_leaving: Why previous employee left
                - candidate_source: How replacement was found
                """)
                df = None
        except Exception as e:
            st.sidebar.error(f"Error processing the uploaded file: {str(e)}")
            st.sidebar.info("Please check the file format and try again")
            df = None

# Check if data is loaded
if df is not None:
    # Add filters to sidebar
    st.sidebar.subheader("Filters")
    
    # Helper function to safely sort values that might contain mixed types
    def safe_sort(values):
        # Convert to strings first to allow sorting mixed types
        try:
            # Try direct sorting first
            return sorted(values)
        except TypeError:
            # If that fails, convert everything to strings
            str_values = [str(x) if not pd.isna(x) else "Not Specified" for x in values]
            return sorted(set(str_values))  # Use set to remove duplicates
    
    # Department filter - convert NaN to "Not Specified"
    df['department'] = df['department'].fillna("Not Specified")
    all_departments = safe_sort(df['department'].unique())
    selected_departments = st.sidebar.multiselect(
        "Select Departments:",
        options=all_departments,
        default=all_departments
    )
    
    # Job role filter - convert NaN to "Not Specified"
    df['job_role'] = df['job_role'].fillna("Not Specified")
    all_roles = safe_sort(df['job_role'].unique())
    selected_roles = st.sidebar.multiselect(
        "Select Job Roles:",
        options=all_roles,
        default=[]  # Default to no filter
    )
    
    # Seniority level filter - convert NaN to "Not Specified"
    df['seniority_level'] = df['seniority_level'].fillna("Not Specified")
    all_seniority = safe_sort(df['seniority_level'].unique())
    selected_seniority = st.sidebar.multiselect(
        "Select Seniority Levels:",
        options=all_seniority,
        default=[]  # Default to no filter
    )
    
    # Reason for leaving filter - convert NaN to "Not Specified"
    df['reason_for_leaving'] = df['reason_for_leaving'].fillna("Not Specified")
    # Convert all reasons to strings to avoid type comparison issues
    df['reason_for_leaving'] = df['reason_for_leaving'].astype(str)
    all_reasons = safe_sort(df['reason_for_leaving'].unique())
    selected_reasons = st.sidebar.multiselect(
        "Select Reasons for Leaving:",
        options=all_reasons,
        default=[]  # Default to no filter
    )
    
    # Time period selector
    st.sidebar.subheader("Time Period")
    
    # Get min and max dates from the dataset
    min_date = df['vacancy_date'].min().date()
    max_date = df['filled_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    
    # Convert to datetime for filtering
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Apply filters
    filtered_df = df.copy()
    
    # Department filter
    if selected_departments:
        filtered_df = filtered_df[filtered_df['department'].isin(selected_departments)]
    
    # Role filter
    if selected_roles:
        filtered_df = filtered_df[filtered_df['job_role'].isin(selected_roles)]
    
    # Seniority filter
    if selected_seniority:
        filtered_df = filtered_df[filtered_df['seniority_level'].isin(selected_seniority)]
    
    # Reason filter
    if selected_reasons:
        filtered_df = filtered_df[filtered_df['reason_for_leaving'].isin(selected_reasons)]
    
    # Date filter - include positions that became vacant or were filled within the selected period
    filtered_df = filtered_df[
        ((filtered_df['vacancy_date'] >= start_date) & (filtered_df['vacancy_date'] <= end_date)) |
        ((filtered_df['filled_date'] >= start_date) & (filtered_df['filled_date'] <= end_date))
    ]
    
    # Check if filtered data exists
    if not filtered_df.empty:
        # -----------------------------------
        # Main dashboard content
        # -----------------------------------
        
        # 1. Key metrics
        st.header("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_fill_time = filtered_df['days_to_fill'].mean()
            st.metric("Average Fill Time", f"{avg_fill_time:.1f} days")
        
        with col2:
            median_fill_time = filtered_df['days_to_fill'].median()
            st.metric("Median Fill Time", f"{median_fill_time:.1f} days")
        
        with col3:
            positions_count = len(filtered_df)
            st.metric("Positions Filled", positions_count)
        
        with col4:
            max_fill_time = filtered_df['days_to_fill'].max()
            st.metric("Longest Fill Time", f"{max_fill_time:.1f} days")
        
        # 2. Fill Time by Department
        st.header("Fill Time Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "By Department", 
            "By Seniority", 
            "By Reason for Leaving"
        ])
        
        with tab1:
            # Department statistics
            dept_stats = filtered_df.groupby('department')['days_to_fill'].agg(['mean', 'median', 'count']).reset_index()
            dept_stats.columns = ['Department', 'Average Days', 'Median Days', 'Positions Filled']
            dept_stats = dept_stats.sort_values('Average Days', ascending=False)
            
            # Create bar chart for departments
            fig_dept = px.bar(
                dept_stats,
                x='Department',
                y='Average Days',
                color='Department',
                hover_data=['Median Days', 'Positions Filled'],
                title='Average Position Fill Time by Department',
                height=500
            )
            
            # Add median line
            for i, row in dept_stats.iterrows():
                fig_dept.add_shape(
                    type='line',
                    x0=i-0.4, x1=i+0.4,
                    y0=row['Median Days'], y1=row['Median Days'],
                    line=dict(color='red', width=3)
                )
            
            # Update layout
            fig_dept.update_layout(
                xaxis_title="Department",
                yaxis_title="Days to Fill Position",
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Add annotation for median
            fig_dept.add_annotation(
                x=1.02, y=1.05,
                text="Red lines indicate median values",
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="left"
            )
            
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with tab2:
            # Seniority statistics
            seniority_stats = filtered_df.groupby('seniority_level')['days_to_fill'].agg(['mean', 'median', 'count']).reset_index()
            seniority_stats.columns = ['Seniority', 'Average Days', 'Median Days', 'Positions Filled']
            
            # Create custom order for seniority
            # Only include levels that exist in our data
            standard_levels = ['Junior', 'Mid-level', 'Senior', 'Executive']
            available_levels = set(seniority_stats['Seniority'])
            
            # Create a list of levels that are in both standard_levels and available_levels
            seniority_order = [level for level in standard_levels if level in available_levels]
            
            # Add any other levels that might be in our data but not in the standard list
            other_levels = [level for level in available_levels if level not in standard_levels]
            seniority_order.extend(sorted(other_levels))
            
            # Apply sorting only if we have valid categories
            if len(seniority_order) > 0:
                try:
                    # Sort dataframe based on custom order
                    seniority_stats['Seniority'] = pd.Categorical(
                        seniority_stats['Seniority'], 
                        categories=seniority_order, 
                        ordered=True
                    )
                    seniority_stats = seniority_stats.sort_values('Seniority')
                except (TypeError, ValueError) as e:
                    # Fall back to sorting by average days if categorical sorting fails
                    seniority_stats = seniority_stats.sort_values('Average Days')
            
            # Create bar chart for seniority
            fig_seniority = px.bar(
                seniority_stats,
                x='Seniority',
                y='Average Days',
                color='Seniority',
                hover_data=['Median Days', 'Positions Filled'],
                title='Average Position Fill Time by Seniority Level',
                height=500
            )
            
            # Add median line
            for i, row in seniority_stats.iterrows():
                fig_seniority.add_shape(
                    type='line',
                    x0=i-0.4, x1=i+0.4,
                    y0=row['Median Days'], y1=row['Median Days'],
                    line=dict(color='red', width=3)
                )
            
            # Update layout
            fig_seniority.update_layout(
                xaxis_title="Seniority Level",
                yaxis_title="Days to Fill Position",
                showlegend=False,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_seniority, use_container_width=True)
        
        with tab3:
            # Reason statistics
            reason_stats = filtered_df.groupby('reason_for_leaving')['days_to_fill'].agg(['mean', 'median', 'count']).reset_index()
            reason_stats.columns = ['Reason', 'Average Days', 'Median Days', 'Positions Filled']
            reason_stats = reason_stats.sort_values('Average Days', ascending=False)
            
            # Create bar chart for reasons
            fig_reason = px.bar(
                reason_stats,
                x='Reason',
                y='Average Days',
                color='Reason',
                hover_data=['Median Days', 'Positions Filled'],
                title='Average Position Fill Time by Reason for Leaving',
                height=500
            )
            
            # Add median line
            for i, row in reason_stats.iterrows():
                fig_reason.add_shape(
                    type='line',
                    x0=i-0.4, x1=i+0.4,
                    y0=row['Median Days'], y1=row['Median Days'],
                    line=dict(color='red', width=3)
                )
            
            # Update layout
            fig_reason.update_layout(
                xaxis_title="Reason for Leaving",
                yaxis_title="Days to Fill Position",
                showlegend=False,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_reason, use_container_width=True)
        
        # 3. Box plots for distribution
        st.header("Fill Time Distribution Analysis")
        
        # Department box plot
        fig_box_dept = px.box(
            filtered_df,
            x='department',
            y='days_to_fill',
            color='department',
            title='Fill Time Distribution by Department',
            height=500,
            points="all"  # Show all points
        )
        
        fig_box_dept.update_layout(
            xaxis_title="Department",
            yaxis_title="Days to Fill Position",
            showlegend=False,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_box_dept, use_container_width=True)
        
        # 4. Heatmap of fill times
        st.header("Fill Time Heatmap")
        
        # Create pivot table for heatmap
        if len(filtered_df['department'].unique()) > 1 and len(filtered_df['seniority_level'].unique()) > 1:
            heat_pivot = filtered_df.pivot_table(
                values='days_to_fill',
                index='department',
                columns='seniority_level',
                aggfunc='mean'
            )
            
            # Define standard seniority order
            standard_levels = ['Junior', 'Mid-level', 'Senior', 'Executive']
            
            # Get actual columns in the pivot table
            actual_columns = heat_pivot.columns.tolist()
            
            # Create ordered list of columns that exist in both standard_levels and actual_columns
            ordered_columns = [level for level in standard_levels if level in actual_columns]
            
            # Add any remaining columns that aren't in standard_levels
            other_columns = [col for col in actual_columns if col not in standard_levels]
            ordered_columns.extend(sorted(other_columns))
            
            # Reorder columns if we have any to reorder
            if ordered_columns:
                try:
                    heat_pivot = heat_pivot[ordered_columns]
                except Exception:
                    # If reordering fails, just use the original columns
                    pass
            
            # Create heatmap
            fig_heat = px.imshow(
                heat_pivot,
                text_auto='.1f',
                aspect="auto",
                title='Average Fill Time by Department and Seniority Level (Days)',
                labels=dict(x="Seniority Level", y="Department", color="Days to Fill"),
                color_continuous_scale="RdYlGn_r"  # Red for high values (bad), green for low (good)
            )
            
            fig_heat.update_layout(
                height=400,
                coloraxis_colorbar=dict(title="Days")
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Not enough data variety to create a heatmap. Please adjust filters.")
        
        # 5. Fill time trend over time
        st.header("Fill Time Trends")
        
        # Create scatter plot with trend line
        fig_trend = px.scatter(
            filtered_df,
            x='vacancy_date',
            y='days_to_fill',
            color='department',
            size='days_to_fill',
            hover_data=['job_role', 'seniority_level', 'filled_date'],
            title='Position Fill Time Trend',
            height=500,
            trendline='ols'  # Use ordinary least squares instead of lowess
        )
        
        fig_trend.update_layout(
            xaxis_title="Vacancy Date",
            yaxis_title="Days to Fill Position",
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 6. Candidate source analysis
        st.header("Recruitment Channel Analysis")
        
        # Make sure candidate_source is string type
        filtered_df['candidate_source'] = filtered_df['candidate_source'].astype(str)
        
        # Source statistics
        source_stats = filtered_df.groupby('candidate_source')['days_to_fill'].agg(['mean', 'count']).reset_index()
        source_stats.columns = ['Source', 'Average Days', 'Positions Filled']
        source_stats = source_stats.sort_values('Average Days')
        
        # Create horizontal bar chart for source
        fig_source = px.bar(
            source_stats,
            y='Source',
            x='Average Days',
            color='Source',
            text='Average Days',
            orientation='h',
            hover_data=['Positions Filled'],
            title='Average Fill Time by Recruitment Channel',
            height=400
        )
        
        # Update layout
        fig_source.update_layout(
            xaxis_title="Average Days to Fill Position",
            yaxis_title="Recruitment Channel",
            plot_bgcolor='white'
        )
        
        # Update bar text format
        fig_source.update_traces(texttemplate='%{x:.1f}', textposition='outside')
        
        st.plotly_chart(fig_source, use_container_width=True)
        
        # 7. Raw data view (collapsible)
        st.header("Detailed Data")
        
        with st.expander("View Raw Position Fill Time Data"):
            # Display formatted dataframe
            display_df = filtered_df.copy()
            
            # Format dates for display
            display_df['vacancy_date'] = display_df['vacancy_date'].dt.strftime('%Y-%m-%d')
            display_df['filled_date'] = display_df['filled_date'].dt.strftime('%Y-%m-%d')
            
            # Round days to fill to 1 decimal place
            display_df['days_to_fill'] = display_df['days_to_fill'].round(1)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data CSV",
                data=csv,
                file_name="position_fill_times_filtered.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")

else:
    # Display placeholder if data isn't loaded
    st.info("Please load or upload position fill time data to view the dashboard.")

# Footer
st.markdown("---")
st.markdown("Position Fill Time Analysis Dashboard | Developed with Streamlit and Plotly")