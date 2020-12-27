import streamlit as st 
import joblib

import random
import time

model = joblib.load("descision_tree.joblib")

st.title("Predico - The Smart health predictor")
st.header("Please enter your symptoms so that we can help you 🩺 ")
symptoms = st.multiselect('Symptoms:',(),key='symptoms')

st.write(symptoms)

if st.button("Evaluate"):
    with st.spinner('Predicting output...'):
        time.sleep(1)
        st.success("Done!")
        
        # predict = model.predict([[wind_direction(direction),find_month(month),day,time_,speed]])
        # slt.write('Predicted Energy Output (KW/h):', predict.round(2))
        # slt.success('Evaluated!')
        # slt.write("Data collected and evaluated to build model for {} winds".format(direction))
        # slt.image(draw_graph(direction),use_column_width=True)