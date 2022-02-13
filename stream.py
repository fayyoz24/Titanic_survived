from turtle import color, width
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler as scaler
from data_handler import data
from train import training, predict
import plotly.figure_factory as ff


st.markdown("<h1 style='text-align: center; color: blue;'>time to remember TITANIC!...</h1>", unsafe_allow_html=True)
img = Image.open("Titanic.jpg")
st.image(img, width=700)

df = data('./train.csv')

images = st.sidebar.radio('Images', options=('yes', 'no'))
if images == 'yes':


    st.markdown("<h3 style='text-align: center; color: blue;'>The difference berween Male and Female people</h3>", unsafe_allow_html=True)
    img = Image.open("male_female_frist.png")
    st.image(img, width=600)


    st.markdown("<h3 style='text-align: center; color: blue;'>The difference berween Male and Female survived people</h3>", unsafe_allow_html=True)
    img5 = Image.open("second.png")
    st.image(img5, width=800)


    survived_male = df.loc[(df['Survived'] == 1)&(df['Sex'] == 'male')]['Age']
    survived_female = df.loc[(df['Survived'] == 1)&(df['Sex'] == 'female')]['Age']

    hist_data = [survived_male]
    group_labels = ['survived_male']

    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[10, 250])
    fig.update_layout(height=800, width=1200)
    st.plotly_chart(fig, use_container_width=True)


    hist_data = [survived_female]
    group_labels = ['survived_female']

    fig1 = ff.create_distplot(
            hist_data, group_labels, bin_size=[10, 250],colors = ['red'])
    fig1.update_layout(height=800, width=1200)
    st.plotly_chart(fig1, use_container_width=True)


    st.markdown("<h3 style='text-align: center; color: blue;'>Try to choose correct class!</h3>", unsafe_allow_html=True)
    img2 = Image.open('third.png')
    st.image(img2, width=800)
    
    st.markdown("<h3 style='text-align: center; color: blue;'>Choose the correct Port</h3>", unsafe_allow_html=True)
    img3 = Image.open('fourth.png')
    st.image(img3, width=800)


models, treeclassifier =training()

data_ = st.sidebar.radio(
    label="Do you want to see the whole data?",
    options=("No",'Yes'))

if data_ == 'Yes':
    st.markdown("<h1 style='text-align: center; color: white;'>The given data!</h1>", unsafe_allow_html=True)
    st.write(df)

    st.markdown("<h4 style='text-align: center; color: survival - Survival (0 = No; 1 = Yes) </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>name - Name</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>sex - Sex </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>age - Age </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>parch - Number of Parents/Children Aboard </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>fare - Passenger Fare </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>parch - Number of Parents/Children Aboard </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>embarked - Port of Embarkation </h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: blue;'>(C = Cherbourg; Q = Queenstown; S = Southampton)  </h4>", unsafe_allow_html=True)
    

st.sidebar.subheader("Alghorithms")
top_book_ = st.sidebar.selectbox(
    label ="Select your Algorithm",
    options = models.keys())

if top_book_ :
    st.text('''
    
    

     ''')
    st.markdown(f"<h3 style='text-align: center; color: blue;'>{round(models.get(top_book_)*100, 1)} %</h3>", unsafe_allow_html=True)

try: 
    user_input = st.sidebar.radio(
        label="Let's check your chance of surviving if you were in Titanic!",
        options=("No",'Yes'))

    if user_input == 'Yes':

        pclass = st.radio(label='Choose your Class', options=(1,2,3))

        sex_ = st.radio(label='Choose your gender!', options=('male', 'female'))
        
        age = st.text_input("Enter your age!\n")

        parch = st.text_input("Enter the number of your children \n")

        fare = st.text_input("How much is your ticket? \n")

        embarked = st.radio(label='Choose your Port of Embarkation \nC = Cherbourg;\nQ = Queenstown; \nS = Southampton', options=(['S', 'C', 'Q']))

        family = st.text_input("Enter your Family size (siblings+children+yourself) \n")

        x = pd.DataFrame({"Pclass":pclass, "Sex":sex_, 'Age':age, 'Parch':parch, 
                        'Fare':fare, 'Embarked':embarked,'Family_size':family}, index=[0])
      
        predict1 = st.button("predict")
        if predict1:
            pred = predict(x)
            if pred > 0.5:
                st.markdown("<h3 style='text-align: center; color: blue;'>You are Lucky!!!, You can survive!</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: blue;'>Please, stay at home!</h3>", unsafe_allow_html=True)

except ValueError:
    st.error('Please enter valid data type!')

about = st.sidebar.button("contributors")
if about:
    st.title("Fayyozjon Usmonov")
    st.markdown("<h1 style='text-align: center; color: blue;'>Thanks for your attention!</h1>", unsafe_allow_html=True)