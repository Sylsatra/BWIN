import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import os
import pickle

# ------------------------------------------------------------------------------
# 1. Define Model Classes
# ------------------------------------------------------------------------------
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        self.dropout_val = dropout
        self.layers = torch.nn.ModuleList()
        # First layer
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        # Last layer
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_val, training=self.training)
        return x

# ------------------------------------------------------------------------------
# 2. Define Mappings
# ------------------------------------------------------------------------------
frequency_mapping_q1 = {
    "Never": 0,
    "Rarely (1-2 times a week)": 1,
    "Sometimes (3-4 times a week)": 2,
    "Often (5-6 times a week)": 3,
    "Every night": 4
}

frequency_mapping_q2 = {
    "Never": 0,
    "Rarely (1-2 times a week)": 1,
    "Sometimes (3-4 times a week)": 2,
    "Every day": 3,
    "Often (5-6 times a week)": 4
}

miss_classes_mapping = {
    "Never": 0,
    "Rarely (1-2 times a month)": 1,
    "Sometimes (1-2 times a week)": 2,
    "Often (3-4 times a week)": 3,
    "Always": 4
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
    "Below Average": 0,
    "Poor": 1,
    "Average": 2,
    "Good": 3,
    "Excellent": 4,
}

year_mapping = {
    "First year": 1,
    "Second year": 2,
    "Third year": 3,
    "Graduate student": 4
}

gender_mapping = {
    "Male": 0,
    "Female": 1
}

# ------------------------------------------------------------------------------
# 3. Define Survey Questions and Options
# ------------------------------------------------------------------------------
survey_questions = {
    "1. What is your year of study?": list(year_mapping.keys()),
    "2. What is your gender?": list(gender_mapping.keys()),

    # Updated Questions with Specific Answer Options
    "3. How often do you have difficulty falling asleep at night?": list(frequency_mapping_q1.keys()),
    "4. On average, how many hours of sleep do you get on a typical day?": list(hours_mapping.keys()),
    "5. How often do you wake up during the night and have trouble falling back asleep?": list(frequency_mapping_q1.keys()),
    "6. How would you rate the overall quality of your sleep?": list(quality_mapping.keys()),
    "7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?": list(frequency_mapping_q1.keys()),
    "9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?": list(miss_classes_mapping.keys()),
    "10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?": list(impact_mapping.keys()),
    # Updated Questions with Specific Answer Options
    "11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?": list(frequency_mapping_q1.keys()),
    "12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?": list(frequency_mapping_q2.keys()),
    "13. How often do you engage in physical activity or exercise?": list(frequency_mapping_q2.keys()),
    "14. How would you describe your stress levels related to academic workload?": list(stress_mapping.keys()),
}

# ------------------------------------------------------------------------------
# 4. Mapping Function
# ------------------------------------------------------------------------------
def map_responses(responses):
    """
    Maps the user responses to numerical values based on predefined mappings.

    Args:
        responses (dict): Dictionary of user responses.

    Returns:
        mapped_data (dict): Dictionary of mapped numerical values.
    """
    mapped_data = {}
    for question, answer in responses.items():
        if question == "1. What is your year of study?":
            mapped_data["1. What is your year of study?"] = year_mapping.get(answer, np.nan)
        elif question == "2. What is your gender?":
            mapped_data["2. What is your gender?"] = gender_mapping.get(answer, np.nan)
        elif question == "15. How would you rate your overall academic performance (GPA or grades) in the past semester?":
            mapped_data["15. How would you rate your overall academic performance (GPA or grades) in the past semester?"] = performance_mapping.get(answer, np.nan)  # To be excluded during inference
        elif question in [
            "3. How often do you have difficulty falling asleep at night?",
            "5. How often do you wake up during the night and have trouble falling back asleep?",
            "11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?"
        ]:
            if question.startswith("3."):
                mapped_data["3. How often do you have difficulty falling asleep at night?"] = frequency_mapping_q1.get(answer, np.nan)
            elif question.startswith("5."):
                mapped_data["5. How often do you wake up during the night and have trouble falling back asleep?"] = frequency_mapping_q1.get(answer, np.nan)
            elif question.startswith("11."):
                mapped_data["11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?"] = frequency_mapping_q1.get(answer, np.nan)
        elif question == "8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?":
            # Use frequency_mapping_q1 for fatigue_mapping
            mapped_data["8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?"] = frequency_mapping_q1.get(answer, np.nan)  # To be excluded during inference
        elif question == "9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?":
            mapped_data["9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?"] = miss_classes_mapping.get(answer, np.nan)
        elif question in [
            "12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?",
            "13. How often do you engage in physical activity or exercise?"
        ]:
            if question.startswith("12."):
                mapped_data["12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?"] = frequency_mapping_q2.get(answer, np.nan)
            elif question.startswith("13."):
                mapped_data["13. How often do you engage in physical activity or exercise?"] = frequency_mapping_q2.get(answer, np.nan)
        elif question.startswith("4."):
            mapped_data["4. On average, how many hours of sleep do you get on a typical day?"] = hours_mapping.get(answer, np.nan)
        elif question.startswith("6."):
            mapped_data["6. How would you rate the overall quality of your sleep?"] = quality_mapping.get(answer, np.nan)
        elif question.startswith("7."):
            mapped_data["7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?"] = frequency_mapping_q1.get(answer, np.nan)
        elif question.startswith("10."):
            mapped_data["10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?"] = impact_mapping.get(answer, np.nan)
        elif question.startswith("14."):
            mapped_data["14. How would you describe your stress levels related to academic workload?"] = stress_mapping.get(answer, np.nan)
        else:
            mapped_data[question] = np.nan  # Handle unexpected questions
    return mapped_data

# ------------------------------------------------------------------------------
# 5. Graph Construction Function
# ------------------------------------------------------------------------------
def build_topk_graph(x_scaled, k=15):
    """
    Builds a top-K similarity graph based on cosine similarity.

    Args:
        x_scaled (np.ndarray): Scaled feature matrix.
        k (int): Number of top similar neighbors.

    Returns:
        edge_index (torch.LongTensor): Edge indices.
        edge_attr (torch.FloatTensor): Edge weights (normalized).
    """
    sim_matrix = cosine_similarity(x_scaled)
    N = sim_matrix.shape[0]
    edges, wts = [], []
    for i in range(N):
        row = sim_matrix[i]
        # Sort descending
        sorted_idx = np.argsort(-row)
        count = 0
        for idx in sorted_idx:
            if idx == i:
                continue
            edges.append([i, idx])
            wts.append(row[idx])
            count += 1
            if count >= k:
                break
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    edge_wts = torch.tensor(wts, dtype=torch.float)
    return edge_index, edge_wts

def rowwise_normalize(edge_index, edge_attr, num_nodes):
    """
    Normalizes edge attributes row-wise.

    Args:
        edge_index (torch.LongTensor): Edge indices.
        edge_attr (torch.FloatTensor): Edge weights.
        num_nodes (int): Number of nodes.

    Returns:
        normalized_edge_attr (torch.FloatTensor): Normalized edge weights.
    """
    sums = torch.zeros(num_nodes, dtype=torch.float)
    src = edge_index[0]
    sums.index_add_(0, src, edge_attr)
    eps = 1e-12
    normalized_edge_attr = edge_attr / (sums[src] + eps)
    return normalized_edge_attr

# ------------------------------------------------------------------------------
# 6. Model Loading Function
# ------------------------------------------------------------------------------
# Removed @st.cache_resource to prevent caching issues
def load_graphsage_models(seeds, input_dim_sage=13, hidden_dim_sage=256, output_dim=3, num_layers_sage=3):
    """
    Loads GraphSAGE models for all specified seeds.

    Args:
        seeds (list): List of seed integers.
        input_dim_sage (int): Input dimension for GraphSAGE.
        hidden_dim_sage (int): Hidden dimension for GraphSAGE.
        output_dim (int): Output dimension (number of classes).
        num_layers_sage (int): Number of layers for GraphSAGE.

    Returns:
        models (dict): Dictionary mapping seed to loaded GraphSAGE model.
    """
    checkpoint_dir = "checkpoints"
    models = {}
    for seed in seeds:
        model = GraphSAGEModel(in_channels=input_dim_sage, hidden_channels=hidden_dim_sage,
                               out_channels=output_dim, num_layers=num_layers_sage).to('cpu')
        checkpoint_path = os.path.join(checkpoint_dir, f"GraphSAGE_seed{seed}.pth")
        if os.path.exists(checkpoint_path):
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                model.eval()
                models[seed] = model
                # Removed the following line for cleaner UI
                # st.write(f"**Model loaded for seed {seed}.**")
            except Exception as e:
                st.error(f"Error loading model for seed {seed} from {checkpoint_path}: {e}")
        else:
            st.error(f"Checkpoint not found for seed {seed} at {checkpoint_path}")
    return models

# ------------------------------------------------------------------------------
# 7. Data Loading Function
# ------------------------------------------------------------------------------
# Removed @st.cache_data to prevent caching issues
def load_processed_data():
    """
    Loads the processed dataset required for GraphSAGE inference.

    Returns:
        df (pd.DataFrame): Processed DataFrame.
    """
    dataset_path = os.path.join("output", "unscaled_processed_dataset.csv")
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            # Removed the following line for cleaner UI
            # st.write("**Processed dataset loaded successfully.**")
            return df
        except Exception as e:
            st.error(f"Error reading processed dataset at {dataset_path}: {e}")
            return None
    else:
        st.error(f"Processed dataset not found at {dataset_path}. Required for GraphSAGE inference.")
        return None

# ------------------------------------------------------------------------------
# 8. Scaler Loading Function
# ------------------------------------------------------------------------------
# Removed @st.cache_data to prevent caching issues
def load_scaler(seed, drop_fatigue=True):
    """
    Loads the scaler object for feature scaling based on the seed and fatigue drop status.

    Args:
        seed (int): The seed corresponding to the model.
        drop_fatigue (bool): Whether fatigue was dropped during training.

    Returns:
        scaler (sklearn scaler object): Loaded scaler.
    """
    scaler_path = f"checkpoints/scaler_seed{seed}_{'drop' if drop_fatigue else 'keep'}_fatigue.pkl"
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            # Removed the following line for cleaner UI
            # st.write(f"**Scaler loaded for seed {seed}.**")
            return scaler
        except Exception as e:
            st.error(f"Error loading scaler for seed {seed} from {scaler_path}: {e}")
            return None
    else:
        st.error(f"Scaler file not found at {scaler_path}.")
        return None

# ------------------------------------------------------------------------------
# 9. Streamlit App
# ------------------------------------------------------------------------------
def main():
    st.title("Student GPA Prediction Survey")
    st.write("Please fill out the following survey to predict your GPA category.")

    # Initialize a dictionary to store user responses
    user_responses = {}

    # Iterate through the survey questions and create dropdowns
    for question, options in survey_questions.items():
        user_responses[question] = st.selectbox(question, options)

    # Submit Button
    if st.button("Submit"):
        # Map the responses
        mapped_data = map_responses(user_responses)

        # Check for any NaN values
        if any(pd.isna(v) for v in mapped_data.values()):
            st.error("Some responses could not be mapped to numerical values. Please check your inputs.")
        else:
            # Display the mapped data
            st.subheader("Mapped Numerical Inputs")
            mapped_df = pd.DataFrame([mapped_data])
            st.table(mapped_df)

            # Define seeds
            seeds = [42, 100, 2023]
            drop_fatigue = True  # Always drop fatigue

            # Load all GraphSAGE models
            with st.spinner('Loading GraphSAGE models...'):
                models = load_graphsage_models(seeds)

            # Check if any models failed to load
            if not models:
                st.error("No GraphSAGE models were loaded. Please check the checkpoint files.")
                return

            # Prepare input features
            # Define the required features in the correct order (excluding 'Performance' and 'Fatigue_During_Day')
            required_features = [
                "1. What is your year of study?",
                "2. What is your gender?",
                "3. How often do you have difficulty falling asleep at night?",
                "4. On average, how many hours of sleep do you get on a typical day?",
                "5. How often do you wake up during the night and have trouble falling back asleep?",
                "6. How would you rate the overall quality of your sleep?",
                "7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?",
                # Exclude '8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?'
                "9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?",
                "10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?",
                "11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?",
                "12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?",
                "13. How often do you engage in physical activity or exercise?",
                "14. How would you describe your stress levels related to academic workload?"
            ]

            # Extract the required features from mapped_data
            input_features = []
            missing_features = []
            for feat in required_features:
                if feat in mapped_data and not pd.isna(mapped_data[feat]):
                    input_features.append(mapped_data[feat])
                else:
                    missing_features.append(feat)

            if missing_features:
                st.error(f"Missing features: {', '.join(missing_features)}. Please ensure all survey questions are answered correctly.")
                return

            # Convert to numpy array with shape (1,13)
            input_features_array = np.array(input_features, dtype=np.float32).reshape(1, -1)

            # Removed the Input Features Array display for cleaner UI

            # Load the processed dataset for GraphSAGE
            with st.spinner('Loading processed data...'):
                df = load_processed_data()
            if df is None:
                return

            # Ensure that the required features exist in the dataframe
            graphsage_feature_cols = required_features  # Already excludes 'Performance' and 'Fatigue_During_Day'
            missing_cols = [feat for feat in graphsage_feature_cols if feat not in df.columns]
            if missing_cols:
                st.error(f"The following required features are missing in the processed dataset: {', '.join(missing_cols)}")
                return

            # Iterate through each seed to perform scaling, graph construction, and inference
            predictions = []
            for seed in seeds:
                with st.spinner(f'Processing Seed {seed}...'):

                    # Load the corresponding scaler for the current seed
                    scaler = load_scaler(seed, drop_fatigue=True)
                    if scaler is None:
                        st.error(f"Scaler for seed {seed} could not be loaded.")
                        continue  # Skip to the next seed

                    # Extract existing features from the processed dataset
                    x_existing = df[graphsage_feature_cols].values  # Shape: (N,13)

                    # Scale existing features using the loaded scaler
                    try:
                        x_existing_scaled = scaler.transform(x_existing)  # Shape: (N,13)
                    except Exception as e:
                        st.error(f"Error scaling existing data for seed {seed}: {e}")
                        continue

                    # Scale the new input features using the same scaler
                    try:
                        x_new_scaled = scaler.transform(input_features_array)  # Shape: (1,13)
                    except Exception as e:
                        st.error(f"Error scaling input features for seed {seed}: {e}")
                        continue

                    # Combine existing and new scaled data
                    x_combined = np.vstack([x_existing_scaled, x_new_scaled])  # Shape: (N+1,13)

                    # Build the top-K similarity graph
                    edge_index, edge_wts = build_topk_graph(x_combined, k=15)

                    # Normalize edge weights
                    edge_attr_normalized = rowwise_normalize(edge_index, edge_wts, num_nodes=x_combined.shape[0])

                    # Create Data object for PyTorch Geometric
                    x_tensor = torch.tensor(x_combined, dtype=torch.float)
                    edge_index_tensor = edge_index  # Already a torch.LongTensor
                    edge_attr_tensor = edge_attr_normalized  # Already a torch.FloatTensor
                    data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

                    # Perform inference with the loaded model
                    model = models.get(seed)
                    if model is None:
                        st.error(f"Model for seed {seed} is not loaded.")
                        continue

                    try:
                        with torch.no_grad():
                            output = model(data.x, data.edge_index, data.edge_attr)
                            logits = output[-1].unsqueeze(0)  # Assuming the new node is the last node
                            probabilities = F.softmax(logits, dim=1).numpy()[0]
                            predicted_class = np.argmax(probabilities)
                            class_mapping = {0: "Below Average", 1: "Average", 2: "High"}
                            predicted_gpa = class_mapping.get(predicted_class, "Unknown")
                            predictions.append({
                                "Seed": seed,
                                "Predicted GPA Category": predicted_gpa,
                                "Below Average Probability (%)": f"{probabilities[0]*100:.2f}",
                                "Average Probability (%)": f"{probabilities[1]*100:.2f}",
                                "High Probability (%)": f"{probabilities[2]*100:.2f}"
                            })

                            # Removed the following lines for cleaner UI
                            # st.write(f"**Seed {seed} - Probabilities:** {probabilities}")
                            # st.write(f"**Seed {seed} - Predicted Class:** {predicted_gpa}")
                    except Exception as e:
                        st.error(f"Error during inference for seed {seed}: {e}")
                        continue

            # Display predictions from all seeds
            if predictions:
                st.subheader("GraphSAGE Model Predictions")
                predictions_df = pd.DataFrame(predictions)
                st.table(predictions_df)
            else:
                st.warning("No predictions were made. Please check the model and scaler checkpoints.")

if __name__ == "__main__":
    main()
