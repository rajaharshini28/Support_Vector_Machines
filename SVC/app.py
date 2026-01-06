import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix

def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved=False

#Folder Setup

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEANED_DIR=os.path.join(BASE_DIR,"data","cleaned")
os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEANED_DIR,exist_ok=True)
log("Application started")
log(f"RAW_DIR={RAW_DIR}")
log(f"CLEANED_DIR = {CLEANED_DIR}")

st.set_page_config("END-TO-END SVM",layout="wide")
st.title("END-TO-END SVM platform")

st.sidebar.header("SVM Settings")
kernel=st.sidebar.selectbox("Kernel",["linear","rbf","poly","sigmoid"])
C=st.sidebar.slider("C (Regularization)", 0.01,10.0,1.0)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVM settings ---> Kernel ={kernel},c={C},Gamma={gamma}")


#Step 1: Data Ingestion

st.header("Step 1 :Data ingestion")
log("Step 1 : Data Ingestion")

option=st.radio("Choose Data Source",["Download Dataset", "Upload CSV"])

df=None
raw_path=None

if option == "Download Dataset":
    if st.button("Download Iris Data"):
        log("Downloading Iris Dataset")
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response=requests.get(url)

        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)

        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded successfully")
        log(f"Iris dataset saved at {raw_path}")

if option == "Upload CSV":
    uploaded_file=st.file_uploader("upload csv file",type=["csv"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        df=pd.read_csv(raw_path)
        st.success("file Uploaded successfully")
        log(f"Uploaded data saved at {raw_path}")


#Step 2: EDA

if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    log("Step 2 Started:EDA")

    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Missing Values:",df.isnull().sum())

    fig,ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = "coolwarm", ax = ax)
    st.pyplot(fig)
    log("EDA completed")

# Step 3 : Data Cleaning

if df is not None:
    st.header("Step 3:Data Cleaning")
    strategy=st.selectbox(
        "Missing value strategy",
        ["Mean","Median","Drop Rows"]
    )

    df_clean=df.copy()

    if strategy=="Drop Rows":
        df_clean=df_clean.dropna()
    
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col]=df_clean[col].fillna(df_clean[col].median())

    st.session_state.df_clean=df_clean
    st.success("Data Cleaning completed")

else:
    st.info("Please Complete Step 1 (Data Ingestion)")


#Step 4:Save Cleaned Data

if st.button("Save cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No Cleaned data found please complete step 3 first")

    else:
        timestamp=datetime.now().strftime("%y%m%d_%H%M%S")
        clean_filename=f"cleaned_dataset_{timestamp}.csv"
        clean_path=os.path.join(CLEANED_DIR,clean_filename)

        st.session_state.df_clean.to_csv(clean_path,index=False)
        st.success("Cleaned Dataset saved")
        st.info(f"saved at : {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")


#Step 5:Load Cleaned Dataset

st.header("Step 5:Load Cleaned Dataset")
clean_files=os.listdir(CLEANED_DIR)
if not clean_files:
    st.warning("No cleaned dataset found please save one in Step 4")
    log("No cleaned dataset available in the terminal")

else:
    selected=st.selectbox("Select Cleaned dataset",clean_files)
    df_model=pd.read_csv(os.path.join(CLEANED_DIR,selected))

    st.success(f"Loaded dataset:{selected}")
    log(f"Loaded clean dataset {selected}")
    st.dataframe(df_model.head())

#Step 6:Train SVM

st.header("Step 6:Train SVM")
log("Step 6 started : SVM Training")

target =st.selectbox("Select Target Column",df_model.columns)
y=df_model[target]
if y.dtype=="object":
    y=LabelEncoder().fit_transform(y)
    log("Target column encoded")

#Select numeric features only

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available for the training")
    st.stop()


#Scale features
scaler=StandardScaler()
x=scaler.fit_transform(x)

#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#Evaluate 
model=SVC(kernel=kernel,C=C,gamma=gamma)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)

st.success(f"Accuracy : {acc:.2f}")
log(f"SVM trained successfully| Accuracy={acc:.2f}")
cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
st.pyplot(fig)