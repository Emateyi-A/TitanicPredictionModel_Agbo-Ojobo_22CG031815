import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


@st.cache_resource
def load_model_and_scaler():
    model = TitanicModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler


model, scaler = load_model_and_scaler()


st.title(" A Titanic Survival Prediction Model System")
st.markdown("Please Enter the  passenger details to predict their survival probability")

column, column_2, column_3 = st.columns(3)

with column:
    age = st.number_input("Age", min_value=0.0, value=25.0, step=0.1)

with column_2:
    sex = st.number_input("Sex (0=Male, 1=Female)", min_value=0.0, max_value=1.0, value=0.0, step=1.0)

with column_3:
    fare = st.number_input("Fare (£)", min_value=0.0, value=50.0, step=1.0)

if st.button("Predict the  Survival", use_container_width=True):
    
    in_data = np.array([[age, sex, fare]])
    in_norma = scaler.transform(in_data)
    
   
    input_tensor = torch.FloatTensor(in_norma)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
   
    st.markdown("---")
    column_1, column_2 = st.columns(2)
    
    with column_1:
        if prediction > 0.5:
            st.success("They  SURVIVED yayy")
            st.write(f"the Confidence: {prediction*100:.2f}%")
        else:
            st.error("They infact did not SURVIVE oops")
            st.write(f"the Confidence: {(1-prediction)*100:.2f}%")
    
    with column_2:
        st.metric("The Survival Probability", f"{prediction:.4f}")
