# 📊 Employee Attrition & Recruitment Analytics Dashboard

An interactive Streamlit dashboard that empowers HR teams to analyze and optimize recruitment performance. It reveals how long it takes to fill open positions after an employee leaves — across departments, roles, and recruitment channels.

---

## 🔍 What It Does

This dashboard visualizes **Position Fill Time Analysis**, helping answer key questions like:

- ⏱️ How long does it typically take to fill a position?
- 🏢 Which departments or job roles have the longest hiring cycles?
- ❗ Are certain reasons for leaving linked to longer hiring times?
- 🧲 Which recruitment sources (e.g., LinkedIn, referrals) are most efficient?

With powerful filtering and visualization, this tool enables data-driven hiring decisions.

---

## ✨ Key Features

- 🎛 **Interactive Filters**  
  Sort by department, job role, experience level, exit reason, and custom date ranges.

- 🔢 **Key Hiring Stats**  
  View metrics like:
  - Average time to fill a position
  - Maximum time to fill
  - Total positions filled

- 📊 **Clear Visualizations**  
  - Bar plots for average fill times by department, seniority, and exit reasons  
  - Box plots showing variability in hiring times across roles  
  - Heatmaps highlighting performance across departments and levels  
  - Trend graphs showing how hiring times evolve over time  
  - Best-performing recruitment channels

- 📁 **Upload Your Data**  
  Analyze your own HR data by uploading a CSV file.

- 📉 **View Raw Data**  
  See the numbers behind the charts and download filtered datasets as CSVs.

---

## 🧰 Tech Stack

| Purpose              | Technology                  |
|----------------------|-----------------------------|
| Web App              | Streamlit                   |
| Data Analysis        | Pandas, NumPy               |
| Data Visualization   | Plotly Express              |
| Dashboard Styling    | Streamlit Widgets           |
| Data Preparation     | Python Scripts              |

---

## 🗂️ Project Structure

Employee-Attrition-Dashboard/
├── app.py # Streamlit app file
├── transform_attrition_data.py # Data transformation script
├── pyproject.toml # Python dependencies
├── requirements.txt # Alternative dependency file
├── data/
│ └── position_fill_times.csv # Processed data for the dashboard
├── attached_assets/
│ └── EmployeeAttrition.csv # Original dataset (optional)
├── generated-icon.jpg # Streamlit app icon (optional)
└── README.md # This documentation


---

## ⚙️ How to Get Started

### ✅ Requirements

- Python 3.8 or newer installed on your system

### 🔁 Step 1: Get the Files

Clone the repo or download the code:

git clone https://github.com/PankajManvani/employee-attrition-dashboard.git
cd employee-attrition-dashboard

🧪 Step 2: Set Up Environment & Install Dependencies
Create and activate a virtual environment:
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Create a requirements.txt with the following content and install:

altair>=5.5.0
folium>=0.19.5
matplotlib>=3.10.1
numpy>=2.2.4
pandas>=2.2.3
plotly>=6.0.1
psycopg2-binary>=2.9.10
python-dotenv>=1.1.0
requests>=2.32.3
scikit-learn>=1.6.1
seaborn>=0.13.2
sqlalchemy>=2.0.40
statsmodels>=0.14.4
streamlit>=1.44.1
streamlit-folium>=0.24.1


pip install -r requirements.txt


🧾 Step 3: Prepare the Dataset (Optional)
If you want to use the default dashboard data:

Place the EmployeeAttrition.csv in the attached_assets/ folder.

Run the transformation script:
python transform_attrition_data.py
This creates data/position_fill_times.csv used by the app.


🚀 How to Run the Dashboard
Once setup is complete, start the app:
streamlit run app.py

It will open in your browser at:
📍 http://localhost:8501

🧭 How to Use the Dashboard
Choose Dataset: Select from the sidebar whether to use the default dataset or upload your own CSV.

Filter View: Apply filters for department, role, experience, exit reason, and date.

Navigate Visuals: Explore trends in each tab to uncover recruitment performance.

Download Data: Export filtered data from the "Raw Data" section.


📦 Data Source
This project is based on the IBM HR Analytics Employee Attrition & Performance dataset.
Data fields include:

Employee ID, Age, Gender, Marital Status

Department, Job Role, Education Level

Attrition Reason, Hire Date, Exit Date

Recruitment Source, Position Fill Time
