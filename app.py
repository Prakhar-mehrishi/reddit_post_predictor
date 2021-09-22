# pipreqs --encoding=utf8  --debug  "path/to/project"
import streamlit as st
from model import predict_pop, small,medium,large
from text_processing import t,d
from datetime import datetime as dt
import pandas as pd

st.set_page_config(page_title="Reddit Text-Post Popularity Prediction App",
                   page_icon="🤖", layout="wide")

feat_cols = ['sub_reddit', 'gilded', 'BinarisedNum_Comments', 
# 'Binarised_is_self',
       'Binarised_over_18', 'polarity', "Title", "Subtext"]

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

with st.form("prediction_form"):
    st.header("Enter the Details about your Reddit Post:")

    Title = st.text_input("Post Title")
    Subtext = st.text_area("Sub-text")
    subreddit = st.selectbox("Choose the subreddit:",
                                small+medium+large) 
    weekday = st.selectbox("Day of Post: ", format_func= lambda x: days[x], options=list(range(len(days))))
    hour = st.time_input("Time of post: [12AM - 12PM]") 
    gilded = st.number_input("Gilded/Gold Awards recieved", value=0, format="%d")
    BinarisedNum_Comments = st.number_input("Number of comments on your post",value=0, format="%d"  )
    Binarised_over_18 = st.radio("Is your post 18+?", format_func= lambda x: "Yes" if x else "No", options=[0, 1])
    domain = st.selectbox("Select your Domain: (from your url)", d)
    thumbnail = st.selectbox("Select Post Thumbnail: ", t)
    

    submit_val = st.form_submit_button("Predict Popularity")



if submit_val:
    attributes = [subreddit, gilded, BinarisedNum_Comments, Binarised_over_18, Title, Subtext, weekday, hour,
                    domain, thumbnail]
    
    result = predict_pop(attributes=attributes)

    if result:
        status = "Popular"
    else:
        status = "Unpopular"

    st.success(f"Your post will be {status}")

    

