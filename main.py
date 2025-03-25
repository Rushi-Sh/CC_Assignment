import streamlit as st
from PIL import Image
from model import predict_disease  # Assuming the model script is in a separate file

st.title("🌱 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_image, detected_diseases, inference_time, detection_info = predict_disease(image)

    st.image(processed_image, caption="Detected Disease", use_container_width=True)

    st.subheader("🌿 Detection Details:")
    st.write(f"**Total Detections:** {len(detected_diseases)}")
    st.write(f"**Inference Time:** {inference_time} sec")

    if detected_diseases:
        for info in detection_info:
            st.markdown(f"""
            - **Label:** {info['Label']}
            - **Class Index:** {info['Class Index']}
            - **Confidence:** {info['Confidence']}
            - **Bounding Box:** {info['Bounding Box']}
            """)
    else:
        st.success("✅ No disease detected.")
