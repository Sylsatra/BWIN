# Repository: Sleep and Academic Performance

Welcome to the **Sleep and Academic Performance** repository! This repository contains the **original CSV dataset** and **Python scripts** needed to encode survey data, run experiments with various Graph Neural Network models (GraphSAGE, GCN, MLP), and test the impact of dropping different features (survey questions) on model performance. You’ll also find a Streamlit app (`survey.py`) that provides an interactive survey interface.

---

## 1. Purpose of the Repository

1. **CSV Data**  
   The repository hosts the original dataset file (`Student_Insomnia_Educational_Outcomes_Dataset.csv`), which includes students’ responses about their sleep habits and academic performance.

2. **Reproducible Experiment Code**  
   Multiple Python scripts are provided to:
   - Encode the survey data into numerical representations.
   - Train and evaluate machine learning models (GraphSAGE, GCN, MLP).
   - Test the impact of dropping certain survey columns (features) on the models’ predictive accuracy.

By following these steps, you can **reproduce** the entire experimentation pipeline on your local machine or in a cloud environment (e.g., Google Colab).

---

## 2. Setting Up the Environment
If you want to use github codespace to run the scripts, follow these steps:
```bash
pip install -r requirements_codespace.txt
```
Note: Training on the CPU may yield different results!

If you want to run locally, follow these steps:
**Required:**
- **Python 3.10.11** (verified to work with pinned dependencies in `requirements.txt`)
- **CUDA 12.4** (for GPU-accelerated PyTorch operations, if applicable)

**Recommended Steps:**

1. **Create a Virtual Environment**  
   ```bash
   python -m venv venv
   ```
   - This command creates a virtual environment named `venv` in the current directory.  
   - Activate it by running:
     - On Windows: `venv\Scripts\activate`
     - On macOS/Linux: `source venv/bin/activate`

2. **Install the Required Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   - This installs all necessary libraries (e.g., PyTorch, PyTorch Geometric, Streamlit, etc.) needed to run the scripts.
   - Ensure your system or container has **CUDA 12.4** installed if you intend to leverage GPU acceleration.

---

## 3. Files and How to Use Them

Below is a brief overview of each file and how to run it:

### **Dataset**

- **`Student_Insomnia_Educational_Outcomes_Dataset.csv`**  
  This is the **original survey dataset** containing information on students’ sleep habits, academic performance, and related behaviors.

### **Encoding & Preprocessing**

- **`process.py`**  
  - **Purpose:** Encodes the survey data into numerical values, handling categorical mappings (e.g., frequency of sleep-related activities).
  - **Usage:**
    ```bash
    python process.py
    ```
  - **Outcome:** Generates a processed dataset (e.g., `unscaled_processed_dataset.csv` in the `output` directory).

### **Model Training & Experimentation**

1. **`train_gnn_domain_param_search.py`**  
   - **Purpose:** Finds the best hyperparameter settings for **GraphSAGE**, **MLP**, and **GCN** by running a domain-based parameter search.
   - **Usage:**
     ```bash
     python train_gnn_domain_param_search.py
     ```
   - **Outcome:** Logs and stores the best configurations, which can be used in subsequent training scripts.

2. **`main.py`**  
   - **Purpose:** Trains **GraphSAGE** and **MLP** models using the best configuration found by `train_gnn_domain_param_search.py`.
   - **Usage:**
     ```bash
     python main.py
     ```
   - **Outcome:** Produces final training results, model checkpoints, and evaluation metrics for GraphSAGE and MLP.

### **Dropping Specific Columns**

These scripts each remove a particular survey question (feature) from the dataset, then re-run the model training to observe how dropping that feature impacts model performance:

- **`drop_class.py`**  
  Drops **“How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?”**

- **`drop_coffee.py`**  
  Drops **“How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?”**

- **`drop_concentrate.py`**  
  Drops **“How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?”**

- **`drop_deadline.py`**  
  Drops **“How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?”**

- **`drop_difficult.py`**  
  Drops **“How often do you have difficulty falling asleep at night?”**

- **`drop_electronic.py`**  
  Drops **“How often do you use electronic devices (e.g., phone, computer) before going to sleep?”**

- **`drop_fatigue.py`**  
  Drops **“How often do you feel fatigued during the day, affecting your ability to study or attend classes?”**

- **`drop_gender.py`**  
  Drops **“What is your gender?”**

- **`drop_hours_of_sleep.py`**  
  Drops **“On average, how many hours of sleep do you get on a typical day?”**

- **`drop_physical.py`**  
  Drops **“How often do you engage in physical activity or exercise?”**

- **`drop_sleep_quality.py`**  
  Drops **“How would you rate the overall quality of your sleep?”**

- **`drop_stress.py`**  
  Drops **“How would you describe your stress levels related to academic workload?”**

- **`drop_wake.py`**  
  Drops **“How often do you wake up during the night and have trouble falling back asleep?”**

**Usage for Each Script**  
```bash
python drop_class.py
```
*(Replace `drop_class.py` with the relevant file name for the feature you wish to remove.)*

**Outcome:**  
These scripts will create a modified dataset (e.g., dropping a specific column) and then typically run the training/evaluation steps, allowing you to see how removing that feature affects model performance.

### **Survey Application**

- **`survey.py`**  
  - **Purpose:** Launches a **Streamlit** app that lets the user fill out an interactive survey. The app then predicts the student’s GPA class (Below Average, Average, High) based on the input data.
  - **Usage:**
    ```bash
    streamlit run survey.py
    ```
  - **Outcome:** Opens a local web browser interface where you can select answers from dropdowns. The script infers GPA class using the trained model.

---

## 4. Github Codespace

If you prefer running these experiments on Github Codespace, you can do so by opening the following link and installing dependencies in `requirements_codespace.txt` :

\[ **[Link to this repository GitHub codespace](https://expert-space-pancake-jx645w6p5vr25pg4.github.dev/)** \]

---

### **Questions or Issues?**

- For any issues or questions, feel free to open an **issue** on this repository, or contact the repository owner directly.

Thank you for your interest in the **Sleep and Academic Performance** repository. We hope you find these scripts and data useful for your research or educational endeavors!
