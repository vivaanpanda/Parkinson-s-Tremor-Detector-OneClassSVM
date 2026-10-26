import streamlit as st
import pandas as pd

columns = ["time", "ax", "ay", "az"]
file = st.file_uploader
if file is not None:
    file_read = pd.read_csv[file]
    file_convert = pd.to_excel(file_read, columns = columns)
    st.download_button(file_convert)