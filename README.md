📢 Machine Failure Analysis Dashboard

📝 Overview

This project is a Streamlit-based Machine Failure Analysis Dashboard that uses Anomaly Detection and Failure Risk Classification techniques to analyze sensor data from different machine types. The dashboard provides interactive visualizations, anomaly detection insights, and failure risk predictions to help prevent unexpected failures.

🚀 Features

📊 Interactive Data Visualization

Line Charts, Bar Charts, Scatter Plots, and Pie Charts.

Correlation Heatmaps and Histogram Distributions.

🔍 Anomaly Detection

Implements Isolation Forest to detect anomalies in machine sensor data.

Visualizes anomalies with color-coded scatter plots.

⚙ Machine Failure Risk Classification

Uses predefined failure risk labels (0 = No Failure, 1 = Failure Risk).

Highlights failure risks in bar charts.

🛠 Machine Type Selection

Filters analysis based on machine types (Drill, Mill, Lathe).

📂 Data Handling & Processing

Handles missing values.

Normalizes numerical sensor readings.

One-hot encodes machine types.

📌 Technologies Used

Python 🐍

Streamlit (For UI & Dashboard) 🎨

Pandas (Data Handling) 📊

Scikit-learn (Machine Learning - Isolation Forest) 🤖

Plotly & Seaborn (Data Visualization) 📈

Matplotlib (Histogram Plots) 🖼

🏗 Setup & Installation

1️⃣ Clone the Repository

git clone https://github.com/SaiyamChopra/Machine-Failure-Analysis.git
cd Machine-Failure-Analysis

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run Machine_Failure_Analysis.py

📁 Project Structure

📂 Machine-Failure-Analysis
│-- Machine_Failure_Analysis.py  # Main Streamlit Dashboard Code
│-- machine_failure_dataset.csv  # Dataset (Ensure it is available in the directory)
│-- README.md                    # Project Documentation
│-- requirements.txt              # Python Dependencies

🛠 Configuration & Customization

Modify machine_failure_dataset.csv to use your own dataset.

Adjust anomaly detection parameters (e.g., contamination in Isolation Forest).

Customize visualizations using Plotly, Seaborn, or Matplotlib.

🔗 Contribution & Version Control

This project is managed using Git & GitHub.

Use feature branches for new enhancements.

Follow best practices for commit messages.

Submit pull requests for any major changes.

📜 License

This project is open-source and available under the MIT License.

📬 Contact

For any queries, reach out via GitHub Issues.
