import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json


# Load JSON file once
with open("treatments.json", "r") as file:
    disease_data = json.load(file)


def get_treatment(disease_name):
    disease = disease_data.get(disease_name)

    if disease:
        return disease
    else:
        return {
            "error": "Disease not found"
        }


#Tensorflow Model Prediction
def confidence_to_severity(confidence):
    if confidence < 0.5:
        return "Low"
    if confidence < 0.75:
        return "Medium"
    return "High"


def load_image_array(test_image):
    test_image.seek(0)
    image = tf.keras.preprocessing.image.load_img(
        test_image,
        target_size=(128, 128)
    )
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    return image_arr / 255.0


def generate_heatmap_images(image_arr):
    attention_map = np.mean(image_arr, axis=-1)
    attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
    attention_map = cv2.normalize(
        attention_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    attention_map = attention_map.astype(np.uint8)

    heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = (image_arr * 255).astype(np.uint8)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return original, superimposed


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    confidence = float(np.max(predictions))
    return np.argmax(predictions), confidence

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        if not test_image:
            st.warning("Please upload an image first.")
            st.stop()
        st.snow()
        st.write("Our Prediction")
        result_index, confidence = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_class = class_name[result_index]
        st.success("Model is Prediction:  {}".format(predicted_class))

        treatment_info = get_treatment(predicted_class)
        if "error" in treatment_info:
            st.warning(treatment_info["error"])
        else:
            st.subheader("Treatment Recommendations")
            st.write("**Disease Name:** {}".format(treatment_info.get("disease_name", predicted_class)))
            st.write(treatment_info.get("description", ""))

            treatment_steps = treatment_info.get("treatment", [])
            if treatment_steps:
                st.markdown("**Treatment:**")
                for step in treatment_steps:
                    st.markdown("- {}".format(step))

            prevention_steps = treatment_info.get("prevention", [])
            if prevention_steps:
                st.markdown("**Prevention:**")
                for step in prevention_steps:
                    st.markdown("- {}".format(step))

        severity = confidence_to_severity(confidence)
        st.subheader("Severity of the Disease")
        st.write(severity)

        image_arr = load_image_array(test_image)
        original, superimposed = generate_heatmap_images(image_arr)

        st.subheader("Attention Heatmap")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                original,
                caption="Original",
                use_column_width=True
            )
        with col2:
            st.image(
                superimposed,
                caption="Heatmap Overlay",
                use_column_width=True
            )
 