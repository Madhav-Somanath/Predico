import streamlit as st 
import joblib

import random
import time

model = joblib.load("lasso.pk1")

st.title("WIND ENERY OUTPUT PREDICTOR")
st.header("PARAMETERS")
month = st.multiselect('Month',('January','Febraury','March','April','May','June','July','August','September','October','November','December'),key='month')

st.write(month)
if st.button("Evaluate"):
    with st.spinner('Predicting output...'):
        time.sleep(1)
        st.error("Done!")
        # predict = model.predict([[wind_direction(direction),find_month(month),day,time_,speed]])
        # slt.write('Predicted Energy Output (KW/h):', predict.round(2))
        # slt.success('Evaluated!')
        # slt.write("Data collected and evaluated to build model for {} winds".format(direction))
        # slt.image(draw_graph(direction),use_column_width=True)