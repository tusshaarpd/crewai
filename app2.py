import streamlit as st
from agents import generate_sales_pitch

st.set_page_config(page_title="🧠 AI Sales Pitch Generator", layout="centered")
st.title("💼 AI-Powered Sales Crew")
st.markdown("Describe your sales lead or inquiry and get a polished pitch in return!")

user_input = st.text_area("📝 Enter the sales inquiry or context", height=150)

if st.button("Generate Pitch"):
    with st.spinner("🚀 Generating pitch..."):
        result = generate_sales_pitch(user_input)
        st.success("✅ Final Sales Pitch")
        st.write(result)
