import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io
import time # For simulating model inference time

# --- 1. CONFIGURATION AND MODEL SIMULATION ---

# Define the set of possible fish categories (simulated classes)
FISH_CATEGORIES = [
    "Salmon",
    "Tuna",
    "Cod",
    "Tilapia",
    "Catfish",
    "Pufferfish",
    "Clownfish",
    "Sardine"
]

def simulate_prediction(image: Image.Image) -> (str, pd.DataFrame):
    """
    Simulates a fish image classification prediction.

    In a real application, this function would load a model (e.g., Keras, PyTorch),
    preprocess the image, and perform model.predict().

    Args:
        image: The uploaded image as a PIL Image object.

    Returns:
        A tuple containing:
        - The predicted category (str).
        - A DataFrame of all category scores (pd.DataFrame).
    """
    # 1. Simulate Inference Delay
    st.info("Analyzing image and running simulated model inference...")
    time.sleep(2) # Simulate network/model processing time

    # 2. Randomly select the "true" predicted category
    np.random.seed(int(time.time() * 1000) % 1000) # Use time for more varied randomness
    predicted_category = np.random.choice(FISH_CATEGORIES)

    # 3. Generate random confidence scores
    num_categories = len(FISH_CATEGORIES)

    # Generate scores where the predicted category has the highest confidence
    scores = np.random.rand(num_categories) * 0.4 # Baseline low scores

    # Boost the predicted category's score significantly
    predicted_index = FISH_CATEGORIES.index(predicted_category)
    scores[predicted_index] += np.random.uniform(0.5, 0.6) # Add 50-60% confidence

    # Normalize scores to ensure they sum up to roughly 100% (and represent probabilities)
    scores = scores / np.sum(scores)

    # Convert to percentage and create DataFrame
    scores_percent = (scores * 100).round(2)

    # Create DataFrame for visualization
    df = pd.DataFrame({
        "Category": FISH_CATEGORIES,
        "Confidence (%)": scores_percent
    })

    # Sort by confidence for better display
    df = df.sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)

    return predicted_category, df


# --- 2. STREAMLIT APPLICATION LAYOUT ---

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Fish Image Classifier",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.title("🐟 Fish Category Classifier")
    st.markdown("Upload an image of a fish below to get a category prediction and confidence scores.")

    # --- File Uploader Widget ---
    uploaded_file = st.file_uploader(
        "Choose a fish image...",
        type=["jpg", "jpeg", "png"],
        help="Only JPEG and PNG images are supported."
    )

    if uploaded_file is not None:
        try:
            # Open the uploaded image
            image = Image.open(uploaded_file)

            # Create two columns for side-by-side display
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Uploaded Image")
                # Display the image
                st.image(image, caption=f"File: {uploaded_file.name}", use_column_width=True)

            with col2:
                # --- Prediction Logic ---
                st.subheader("Classification Result")

                # Check if the "Predict" button is clicked
                if st.button("Run Prediction", use_container_width=True):
                    # Call the simulation function
                    predicted_class, confidence_df = simulate_prediction(image)

                    # Display the final prediction
                    st.markdown("---")
                    st.success(f"**PREDICTION:** {predicted_class}")
                    st.metric(
                        label="Top Confidence Score",
                        value=f"{confidence_df.iloc[0]['Confidence (%)']}%"
                    )

                    st.markdown("---")
                    st.subheader("All Model Confidence Scores")

                    # Display the confidence scores as a bar chart
                    # Using the first column of the DataFrame as the index for the chart
                    st.bar_chart(
                        confidence_df.set_index('Category'),
                        height=300
                    )

                    # Optional: display the raw data table
                    with st.expander("Show detailed confidence table"):
                         st.dataframe(confidence_df, use_container_width=True)

                else:
                    st.warning("Click 'Run Prediction' to classify the image.")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
            st.warning("Please ensure the uploaded file is a valid image (JPEG or PNG).")

    else:
        # Display instructions when no file is uploaded
        st.info("Please upload an image to begin the fish classification.")

        # Example categories for user reference
        st.markdown(
            """
            ### Simulated Categories:
            The model simulates classifying images into one of these types:
            Salmon, Tuna, Cod, Tilapia, Catfish, Pufferfish, Clownfish, Sardine.
            """
        )

if __name__ == "__main__":
    main()