import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx

# ------------------------------------------------------------------------------
# 1. Load and Clean Dataset
# ------------------------------------------------------------------------------
file_path = 'Student_Insomnia_Educational_Outcomes_Dataset.csv'
dataset = pd.read_csv(file_path)
dataset.columns = dataset.columns.str.strip()

# ------------------------------------------------------------------------------
# 2. Define Mappings
# ------------------------------------------------------------------------------
frequency_mapping = {
    "Never": 0,                      # 0 times/week
    "Rarely": 1,                     # ~1 time/week (generic Rarely)
    "Rarely (1-2 times a week)": 1.5, # ~1.5 times/week
    "Rarely (1-2 times a month)": 0.3, # ~0.3 times/week (about 1.2 times/month)
    "Sometimes": 3,                  # ~3 times/week (generic Sometimes)
    "Sometimes (1-2 times a week)": 2.5, # ~2.5 times/week
    "Sometimes (3-4 times a week)": 3.5, # ~3.5 times/week
    "Often (3-4 times a week)": 4.5, # ~3–4 times/week, assigned a midpoint
    "Often": 5,                      # ~5 times/week (generic Often)
    "Often (5-6 times a week)": 5.5, # ~5.5 times/week
    "Every night": 7,                # 7 nights/week
    "Every day": 7,                  # 7 days/week (treating "day" similarly to "night")
    "Always": 7,                     # 7 times/week (equivalent to Every night/day)
}

hours_mapping = {
    "Less than 4 hours": 3.5,
    "4-5 hours": 4.5,
    "6-7 hours": 6.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 8.5,
}
quality_mapping = {
    "Very poor": 0,
    "Poor": 1,
    "Average": 2,
    "Good": 3,
    "Very good": 4,
    "Excellent": 5,
}
impact_mapping = {
    "No impact": 0,
    "Minor impact": 1,
    "Moderate impact": 2,
    "Major impact": 3,
    "Severe impact": 4,
}
stress_mapping = {
    "No stress": 0,
    "Low stress": 1,
    "High stress": 2,
    "Extremely high stress": 3,
}
performance_mapping = {
    # 5-class scenario: 0..4
    "Below Average": 0,
    "Poor": 1,
    "Average": 2,
    "Good": 3,
    "Excellent": 4,
}

mappings = {
    "1. What is your year of study?": {
        "First year": 1,
        "Second year": 2,
        "Third year": 3,
        "Graduate student": 4
    },
    "2. What is your gender?": {
        "Male": 0,
        "Female": 1
    },
    "3. How often do you have difficulty falling asleep at night?": frequency_mapping,
    "4. On average, how many hours of sleep do you get on a typical day?": hours_mapping,
    "5. How often do you wake up during the night and have trouble falling back asleep?": frequency_mapping,
    "6. How would you rate the overall quality of your sleep?": quality_mapping,
    "7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?": frequency_mapping,
    "8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?": frequency_mapping,
    "9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?": frequency_mapping,
    "10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?": impact_mapping,
    "11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?": frequency_mapping,
    "12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?": frequency_mapping,
    "13. How often do you engage in physical activity or exercise?": frequency_mapping,
    "14. How would you describe your stress levels related to academic workload?": stress_mapping,
    "15. How would you rate your overall academic performance (GPA or grades) in the past semester?": performance_mapping,
}

# ------------------------------------------------------------------------------
# 3. Check & Apply Mappings with Unmapped Detection
# ------------------------------------------------------------------------------
for col, mapping in mappings.items():
    if col in dataset.columns:
        # Create a temporary series by mapping
        mapped_series = dataset[col].map(mapping)

        # Identify rows where the new mapped series is NaN,
        # but the original column was not NaN → truly unmapped
        unmapped_mask = mapped_series.isna() & dataset[col].notna()
        unmapped_count = unmapped_mask.sum()

        if unmapped_count > 0:
            # Gather unique unmapped textual entries
            unmapped_entries = dataset.loc[unmapped_mask, col].unique()
            raise ValueError(
                f"Column '{col}' has {unmapped_count} unmapped entries. "
                f"Unmapped categories: {list(unmapped_entries)}\n"
                "Please update your mapping dictionary or handle them appropriately."
            )

        # Assign mapped values (no unmapped issues encountered)
        dataset[col] = mapped_series

# Double-check for any remaining NaN
null_sum = dataset.isna().sum()
if null_sum.any():
    print("Warning: Some columns still contain NaN values after mapping:", null_sum[null_sum > 0])

# ------------------------------------------------------------------------------
# 4. Separate Features and Target
# ------------------------------------------------------------------------------
target_col = "15. How would you rate your overall academic performance (GPA or grades) in the past semester?"
features = dataset.drop(columns=[target_col], errors='ignore')
target = dataset[target_col]
# ------------------------------------------------------------------------------
# 5. Save Outputs
# ------------------------------------------------------------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# SSave the fully mapped but unscaled DataFrame
dataset.to_csv(os.path.join(output_dir, "unscaled_processed_dataset.csv"), index=False)

print(f"Data saved in '{output_dir}' directory.")
