###### ---------------------- MAIN COMPONENT 1: IMPORT LIBRARY ---------------------- ######

# Data Manipulation
import numpy as np
import pandas as pd
import datetime, warnings, scipy
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# Streamlit Components
import streamlit as st 
import streamlit.components.v1 as html
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from streamlit_option_menu import option_menu
import base64

# Visualization Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Visualization Images & Wordcloud
from PIL import Image
from wordcloud import WordCloud
import stylecloud as sc

# Visualization Seaborn & Matplotlib
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.backends.backend_agg import RendererAgg

# Machine Learning Components
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor




###### ---------------------- MAIN COMPONENT 2: SIDEBAR, TITLE, & PAGE CONFIGURATION ---------------------- ######

# Set initial page configuration
st.set_page_config(page_title = "JomCapai Dashboard", layout = "wide", initial_sidebar_state = "expanded")
st.markdown(""" <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """, unsafe_allow_html = True)
st.write('<style> div.block-container{padding-top:0rem;} </style>', unsafe_allow_html = True)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html = True)


# Insert tagline as the dashboard main title
st.markdown(""" <style> .title {font-size: 25px ; font-family: 'Cooper Black'; 
    color: #083DE0; background-color: #FDCA55; border-radius: 20%; text-align: center} </style> """, unsafe_allow_html = True)
st.markdown('<p class="title">Performance Dashboard for An Intro Course In Data SC!</p>', unsafe_allow_html = True)


# Choose on whether to access as a student or non-student
col1, col2 = st.columns([1,2])
col1.markdown(""" <style> .student {font-size: 20px ; font-family: 'Comic Sans MS Black'; 
    color: #0E1117; background-color: #E5AAF5; border-radius: 5%; text-align: left} </style> """, unsafe_allow_html = True)
col1.markdown('<p class="student">ARE YOU A STUDENT??</p>', unsafe_allow_html = True)
student_main_option = col2.radio("", ('YES', 'NO'))


# Insert JomCapai logo at the sidebar
with st.sidebar.container():
    logo = Image.open('./image/logo.png')
    st.image(logo, use_column_width = True)


# Set option menu to choose dashboard components at the sidebar
with st.sidebar:
    choose_dashboard = option_menu("Dashboard Menu", ["Application", "User Guide", "Source Code", "About Author"],
                         icons = ['app-indicator', 'journal-text', 'file-earmark-code', 'person-badge'],
                         menu_icon = 'list', default_index = 0,
                         styles = {
                            "container": {"padding": "5!important", "background-color": "#0E1117"},
                            "icon": {"color": "yellow", "font-size": "20px"}, 
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#93919A"},
                            "nav-link-selected": {"background-color": "#F52C2C"}
                            }
                        )




###### ---------------------- MAIN COMPONENT 3: IMPORT DATA & DEFINE FUNCTIONS ---------------------- ######

# Import cleaned datasets from csv files
@st.cache(allow_output_mutation = True)
def load_data_result():
    return pd.read_csv('data_results_cleaned.csv')

@st.cache(allow_output_mutation = True)
def load_data_attendance():
    return pd.read_csv('data_results_attendance_cleaned.csv')

@st.cache(allow_output_mutation = True)
def load_data_sentiment():
    return pd.read_csv('data_sentiment.csv')



# Define function for image display on "Access Denied" for student/non-student, "About Author", "Github Files"
def image_denied():
    image = Image.open(r'.image\denied.png')
    st.image(image, use_column_width = True)

def image_author():
    image = Image.open(r'.image\author.png')
    st.image(image, use_column_width = True)

def image_github():
    image = Image.open(r'.image\github.png')
    st.image(image, use_column_width = True)


# Define function to embed PDF document under "User Guide"
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="750" height="700" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html = True)



# Define function for RF regression model
@st.cache(allow_output_mutation = True)
def train_model(result_predict):
    global scaler
    Y = result_predict['TOTAL MARKS']
    X = result_predict.drop(['TOTAL MARKS'], axis = 1)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    model = RandomForestRegressor(n_estimators = 160, max_depth = 8, random_state = 3).fit(X, Y)
    return model, scaler



# Define function for grades
def grade_func(marks):
    if marks >= 90:
        grade = 'A+' 
    elif marks >= 80 and marks < 90:
        grade = 'A'
    elif marks >= 75 and marks < 80:
        grade = 'A-'
    elif marks >= 70 and marks < 75:
        grade = 'B+'
    elif marks >= 65 and marks < 70:
        grade = 'B'
    elif marks >= 60 and marks < 65:
        grade = 'B-'
    elif marks >= 55 and marks < 60:
        grade = 'C+'
    elif marks >= 50 and marks < 55:
        grade = 'C'
    elif marks >= 45 and marks < 50:
        grade = 'C-'
    elif marks >= 40 and marks < 45:
        grade = 'D+'
    elif marks >= 35 and marks < 40:
        grade = 'D'
    else:
        grade = 'F'
    return grade


# Define function for action based on predicted student's performance
def action_func(marks):
    if marks >= 80:
        action = 'Encourage (Excellent)' 
    elif marks >= 70 and marks < 80:
        action = 'Explore (Average)'
    else:
        action = 'Engage (At-Risk)'
    return action


# Define function for student's roll number
def number_func(num):
    if num < 10:
        num2 = "00" + str(num)
    elif num >= 10 and num < 100:
        num2 = "0" + str(num)
    else:
        num2 = str(num)
    return num2



# Define function for pictogram on star rating sentiments
def rating_func(rating):
    if rating == 5.00:
        image = Image.open(r'.image\rating_500.png')
    elif rating >= 4.75 and rating < 5.00:
        image = Image.open(r'.image\rating_475.png')
    elif rating >= 4.50 and rating < 4.75:
        image = Image.open(r'.image\rating_450.png')
    elif rating >= 4.25 and rating < 4.50:
        image = Image.open(r'.image\rating_425.png')
    elif rating >= 4.00 and rating < 4.25:
        image = Image.open(r'.image\rating_400.png')
    elif rating >= 3.75 and rating < 4.00:
        image = Image.open(r'.image\rating_375.png')
    elif rating >= 3.50 and rating < 3.75:
        image = Image.open(r'.image\rating_350.png')
    elif rating >= 3.25 and rating < 3.50:
        image = Image.open(r'.image\rating_325.png')
    else:
        image = Image.open(r'.image\rating_300.png')
    return image
    


# Define function for sequence numbers based on grades
def grade_seq_func(grade):
    if grade == 'A+':
        num = 1
    elif grade == 'A':
        num = 2
    elif grade == 'A-':
        num = 3
    elif grade == 'B+':
        num = 4
    elif grade == 'B':
        num = 5
    elif grade == 'B-':
        num = 6
    elif grade == 'C+':
        num = 7
    elif grade == 'C':
        num = 8
    elif grade == 'D+':
        num = 9
    elif grade == 'D':
        num = 10
    else:
        num = 11
    return num








##### ------------------ DASHBOARD COMPONENT 1: APPLICATION ------------------ #####
if choose_dashboard == "Application":

    # Set option menu horizontally to choose application components
    choose_app = option_menu(menu_title = None, options = ["Past Results", "Student Corner", "Predict Grades", "Sentiments"],
                         icons = ['clipboard-data', 'people-fill', 'dice-5', 'emoji-smile'],
                         default_index = 0, orientation = "horizontal",
                         styles = {
                            "container": {"padding": "5!important", "background-color": "#080A71"},
                            "icon": {"color": "yellow", "font-size": "17px"}, 
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#7468FB"},
                            "nav-link-selected": {"background-color": "#F52C2C"}
                            }
                        )





    #### ------------ APP COMPONENT 1: PAST RESULTS (Descriptive Analysis On Overall Results) ------------ ####
    if choose_app == "Past Results":
        result = load_data_result()
        attendance = load_data_attendance()


        # Title for big number chart with expander
        st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
            color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
        st.markdown('<p class="subtitle">Lifetime Results</p>', unsafe_allow_html = True)


        choice_expander = st.expander(label = 'Click "+" to choose your desired display of number')
        
        with choice_expander:
            col0A, col0B = st.columns([1,5])
            col0A.markdown(""" <style> .lifetime {font-size: 18px ; font-family: 'Arial'; 
                    color: #FAFAFA; text-align: left} </style> """, unsafe_allow_html = True)
            col0A.markdown('<p class="lifetime">View As:</p>', unsafe_allow_html = True)
            
            lifetime_option = col0B.radio("", ('Absolute Number', 'Percentage'))



        # Big number chart on lifetime results
        col1, col2, col3, col4 = st.columns([2,2,2,1])
        distinction = result['TOTAL MARKS'] >= 80
        passing = result['TOTAL MARKS'] >= 65
        failing = result['TOTAL MARKS'] < 65


        if lifetime_option == 'Absolute Number':
            col1.metric("Total Students", result.shape[0])
            col2.metric("Pass with Distinction", distinction.sum())
            col3.metric("Pass", passing.sum())
            col4.metric("Fail", failing.sum())

        else:
            col1.metric("Total Students", "100%")
            col2.metric("Pass with Distinction", "{}%".format(round(distinction.sum() *100/result.shape[0], 1)))
            col3.metric("Pass", "{}%".format(round(passing.sum() *100/result.shape[0], 1)))
            col4.metric("Fail", "{}%".format(round(failing.sum() *100/result.shape[0], 1)))
        


        # Title for scatter plot
        st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
            color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
        st.markdown('<p class="subtitle">Correlation of Total Marks & Attendance</p>', unsafe_allow_html = True)


        # Scatter plot for total marks vs attendance rate for Batch 9
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x = attendance['Attendance Rate'], y = attendance['TOTAL MARKS'], mode = 'markers', marker_color = '#EB89B5'))
        fig1.add_vrect(x0 = 80, x1 = 110, line_width = 0, fillcolor = "blue", opacity = 0.05)
        fig1.update_traces(mode = 'markers', marker_line_width = 1, marker_size = 10)
        fig1.update_layout(title = {'text': "Semester 1 Session 2021/2022 - Batch [9]", 'x': 0.5, 'xanchor': 'center'},
                        xaxis_title_text = 'Attendance Rate (%)', yaxis_title_text = 'Total Marks (%)')
        
        col5, col6, col7 = st.columns([1,12,1])
        col5.write("")
        col6.plotly_chart(fig1, use_column_width = True)
        col7.write("")



        # Title for area chart
        st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
            color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
        st.markdown('<p class="subtitle">Grades Across Batches</p>', unsafe_allow_html = True)



        # Area chart for grades across batches
        col8, col9 = st.columns([1,7])
        grade_option = col8.selectbox('Grade', ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D+", "D", "F"), index = 1)

        result_01 = result[result['GRADE'] == grade_option].groupby('Batch', as_index = False).agg({"Roll No.": "count"})
        result_01.rename(columns = {'Roll No.' : 'Frequency'}, inplace = True)
        result_01['Batch'] = result_01['Batch'].astype(str)
        

        fig2 = go.Figure()

        if grade_option == 'A+' or grade_option == 'A':
            fig2.add_trace(go.Scatter(x = result_01['Batch'], y = result_01['Frequency'], fill = 'tozeroy', mode = 'markers', marker_color = '#11DA0A'))
            fig2.update_layout(title = {'text': "Excellent Students", 'x': 0.5, 'xanchor': 'center'},
                            xaxis_title_text = 'Batch', yaxis_title_text = 'No. of Students', xaxis = dict(showgrid = False))
        
        elif grade_option == 'A-' or grade_option == 'B+':
            fig2.add_trace(go.Scatter(x = result_01['Batch'], y = result_01['Frequency'], fill = 'tozeroy', mode = 'markers', marker_color = '#F9DB16'))
            fig2.update_layout(title = {'text': "Average Students", 'x': 0.5, 'xanchor': 'center'},
                            xaxis_title_text = 'Batch', yaxis_title_text = 'No. of Students', xaxis = dict(showgrid = False))
        
        else:
            fig2.add_trace(go.Scatter(x = result_01['Batch'], y = result_01['Frequency'], fill = 'tozeroy', mode = 'markers', marker_color = '#F12323'))
            fig2.update_layout(title = {'text': "At-Risk Students", 'x': 0.5, 'xanchor': 'center'},
                            xaxis_title_text = 'Batch', yaxis_title_text = 'No. of Students', xaxis = dict(showgrid = False))
        
        col9.plotly_chart(fig2, use_column_width = True)



        # Title for histogram
        st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
            color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
        st.markdown('<p class="subtitle">Distribution of Marks Based on Assessment Type</p>', unsafe_allow_html = True)



        # Histogram of marks based on each batch and different assessment type with expander
        result_02 = result[['Batch', 'Semester', 'Session', 'TOTAL MARKS', 'COURSEWORK', 'FINALS', 'Assignment', 'Midterm', 'Online Participation', 
                        'Group Project', 'Alternative Assessment 01', 'Alternative Assessment 02']]

        result_02.rename(columns = {'TOTAL MARKS' : 'All Assessments', 'COURSEWORK' : 'Coursework', 'FINALS' : 'Finals'}, inplace = True)
        result_02 = result_02.melt(id_vars = ['Batch', 'Semester', 'Session'], var_name = 'Assessment Type', value_name = 'Marks')
        

        choice_expander2 = st.expander(label = 'Click "+" to choose your desired semester, session, and assessment type')

        with choice_expander2:
            col10, col11, col12 = st.columns([1,2,2])
            sem_option = col10.radio('Semester', pd.unique(result['Semester']))
            session_option = col11.selectbox('Session', pd.unique(result['Session']))
            
            if session_option == "2017/2018" or session_option == "2018/2019" or (sem_option == 1 and session_option == "2019/2020"):
                assessment_option = col12.selectbox('Assessment Type', ("All Assessments", "Coursework", "Finals", "Assignment", "Midterm", "Online Participation", "Group Project"))
            else:
                assessment_option = col12.selectbox('Assessment Type', pd.unique(result_02['Assessment Type']))
            
            filter_01 = result_02[(result_02['Semester'] == sem_option) & (result_02['Session'] == session_option) & (result_02['Assessment Type'] == assessment_option)]


        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x = filter_01['Marks'], xbins = dict(start = 0, end = 100, size = 2), marker_color = '#EB89B5'))
        fig3.add_vline(x = filter_01['Marks'].median(), line_width = 3, line_color = "green", annotation_text = "Median", 
                    annotation_bgcolor = "blue", annotation_position = "left top")

        fig3.update_layout(title = {'text': "{}: Semester {} Session {} - Batch {}".format(assessment_option, sem_option, session_option, pd.unique(filter_01['Batch'])), 
                        'x': 0.5, 'xanchor': 'center'}, xaxis_title_text = 'Marks', yaxis_title_text = 'Frequency', bargap = 0.05, bargroupgap = 0.05)
        st.plotly_chart(fig3, use_column_width = True)



        # Title for data table
        st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
            color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
        st.markdown('<p class="subtitle">Students Detailed Performances from Batch 1 till 9</p>', unsafe_allow_html = True)



        # Data table on students detailed performances from batch 1 till 9 with expander
        if student_main_option == 'NO':
            table_expander = st.expander(label = 'Click "+" to view the table')

            with table_expander:
                result_03 = result[['Batch', 'Semester', 'Session', 'Roll No.', 'Assignment', 'Midterm', 'Online Participation', 
                            'Group Project', 'COURSEWORK',  'Alternative Assessment 01', 'Alternative Assessment 02', 'FINALS',
                            'TOTAL MARKS', 'GRADE']]

                fig4 = GridOptionsBuilder.from_dataframe(result_03)
                fig4.configure_pagination(paginationAutoPageSize = True)
                fig4.configure_side_bar() 
                fig4.configure_selection('multiple', use_checkbox = True, groupSelectsChildren= "Group checkbox select children")
                gridOptions = fig4.build()

                grid_response = AgGrid(result_03, gridOptions = gridOptions, data_return_mode = 'AS_INPUT', 
                                    update_mode = 'MODEL_CHANGED', fit_columns_on_grid_load = False, theme = 'blue', 
                                    enable_enterprise_modules = True, height = 300, width = '100%', reload_data = False)


                # Data table on students detailed performances from batch 1 till 9 (after selection)
                df = grid_response['data']
                selected = grid_response['selected_rows']
                result_04 = pd.DataFrame(selected) 

                try:
                    st.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                        color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
                    st.markdown('<p class="caption">Your confirmed list of students..</p>', unsafe_allow_html = True)

                    fig5 = st.dataframe(result_04[['Semester', 'Session', 'Roll No.', 'COURSEWORK', 'FINALS', 'TOTAL MARKS', 'GRADE']].style.format(
                                                {'COURSEWORK': '{:.1f}', 'FINALS': '{:.1f}', 'TOTAL MARKS': '{:.1f}'}))
                
                except:
                    st.error("No Data To Show!!")


        else:
            image_denied()





    #### ------------ APP COMPONENT 2: STUDENT CORNER (Descriptive Analysis On Individual Results) ------------ ####
    elif choose_app == "Student Corner":
        if student_main_option == 'YES':
            result = load_data_result()


            # Student login with expanders
            st.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
            st.markdown('<p class="caption">Please login with your details..</p>', unsafe_allow_html = True)
            
            login_expander = st.expander(label = 'Click "+" to view more options')
            
            with login_expander:
                col1, col2, col3 = st.columns([1,2,2])
                sem_option = col1.radio('Semester', pd.unique(result['Semester']))
                session_option = col2.selectbox('Session', pd.unique(result['Session']))
            
                filter_01 = result[(result['Semester'] == sem_option) & (result['Session'] == session_option)]
                num_option = col3.selectbox('Roll No.', pd.unique(filter_01['Roll No.']))
            


            # Title for big number chart with expander
            st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
            st.markdown('<p class="subtitle">Your Actual Marks for Batch {}</p>'.format(pd.unique(filter_01['Batch'])), unsafe_allow_html = True)


            choice_expander = st.expander(label = 'Click "+" to choose your desired display of number')
        
            with choice_expander:
                col0A, col0B = st.columns([1,5])
                col0A.markdown(""" <style> .lifetime {font-size: 18px ; font-family: 'Arial'; 
                        color: #FAFAFA; text-align: left} </style> """, unsafe_allow_html = True)
                col0A.markdown('<p class="lifetime">View As:</p>', unsafe_allow_html = True)
                
                individual_option = col0B.radio("", ('Absolute Number', 'Percentage'))



            # Big number chart on individual student performance
            filter_02 = result[(result['Semester'] == sem_option) & (result['Session'] == session_option) & (result['Roll No.'] == num_option)]

            if session_option == "2017/2018" or session_option == "2018/2019" or (sem_option == 1 and session_option == "2019/2020"):
                col4, col5, col6, col7, col8 = st.columns([2,2,3,3,2])
                
                if individual_option == 'Absolute Number':
                    col4.metric("Assignment (10%)", filter_02['Assignment'])
                    col5.metric("Midterm (20%)", filter_02['Midterm'])
                    col6.metric("Online Participation (10%)", filter_02['Online Participation'])
                    col7.metric("Group Project (20%)", filter_02['Group Project'])
                    col8.metric("Finals (40%)", filter_02['FINALS'])
                
                else:
                    col4.metric("Assignment", "{}%".format(round(filter_02['Assignment'].iloc[0] *10)))
                    col5.metric("Midterm", "{}%".format(round(filter_02['Midterm'].iloc[0] *5)))
                    col6.metric("Online Participation", "{}%".format(round(filter_02['Online Participation'].iloc[0] *10)))
                    col7.metric("Group Project", "{}%".format(round(filter_02['Group Project'].iloc[0] *5)))
                    col8.metric("Finals", "{}%".format(round(filter_02['FINALS'].iloc[0] *2.5)))


            else:
                col4, col5, col6, col7 = st.columns(4)

                if individual_option == 'Absolute Number':
                    col4.metric("Assignment (10%)", filter_02['Assignment'])
                    col5.metric("Midterm (20%)", filter_02['Midterm'])
                    col6.metric("Online Participation (10%)", filter_02['Online Participation'])
                    col7.metric("Group Project (20%)", filter_02['Group Project'])
                    col4.metric("Alt Assessment 01 (20%)", filter_02['Alternative Assessment 01'])
                    col5.metric("Alt Assessment 02 (20%)", filter_02['Alternative Assessment 02'])
                    col6.metric("Finals (40%)", filter_02['FINALS'])
                
                else:
                    col4.metric("Assignment", "{}%".format(round(filter_02['Assignment'].iloc[0] *10)))
                    col5.metric("Midterm", "{}%".format(round(filter_02['Midterm'].iloc[0] *5)))
                    col6.metric("Online Participation", "{}%".format(round(filter_02['Online Participation'].iloc[0] *10)))
                    col7.metric("Group Project", "{}%".format(round(filter_02['Group Project'].iloc[0] *5)))
                    col4.metric("Alt Assessment 01", "{}%".format(round(filter_02['Alternative Assessment 01'].iloc[0] *5)))
                    col5.metric("Alt Assessment 02", "{}%".format(round(filter_02['Alternative Assessment 02'].iloc[0] *5)))
                    col6.metric("Finals", "{}%".format(round(filter_02['FINALS'].iloc[0] *2.5)))
            


            # Title and caption for bullet chart
            st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
            st.markdown('<p class="subtitle">Target Setting for Self-Improvement</p>', unsafe_allow_html = True)

            col9, col10 = st.columns([1,4])
            col9.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
            col9.markdown('<p class="caption">Set your target..</p>', unsafe_allow_html = True)



            # Bullet chart on target setting
            result_03 = result[['Semester', 'Session', 'Batch', 'Roll No.', 'TOTAL MARKS']]
            filter_03 = result_03[(result_03['Semester'] == sem_option) & (result_03['Session'] == session_option) & (result_03['Roll No.'] == num_option)]
            target_input = col10.slider("", 65, 100, 70)

            col11, col12 = st.columns([2,7])
            fig1 = go.Figure(go.Indicator(
                    value = filter_03['TOTAL MARKS'].iloc[0], domain = {'x': [0.1, 1], 'y': [0, 1]}, 
                    title = {'text' :"<b>Total Marks</b>"}, mode = "number+gauge", 
                    gauge = {'shape': "bullet", 'axis': {'range': [None, 100]}, 'bar': {'color': "blue"},
                            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 1, 'value': target_input},
                            'steps': [{'range': [0, 65], 'color': "pink"}, {'range': [65, 80], 'color': "lightgreen"}]}))
            
            fig1.update_layout(height = 250)
            col12.plotly_chart(fig1, use_column_width = True)
            

            # Statement based on chosen target input
            if target_input > filter_03['TOTAL MARKS'].iloc[0]:
                col11.markdown(""" <style> .message {font-size: 15px ; font-family: 'Arial'; 
                            color: #FAFAFA; background-color: #A81313; text-align: center} </style> """, unsafe_allow_html = True)
                col11.markdown('<p class="message">You are still behind by {}% from your target!</p>'.format(round(target_input - filter_03['TOTAL MARKS'].iloc[0], 1)), unsafe_allow_html = True)
            
            else:
                col11.markdown(""" <style> .message {font-size: 16px ; font-family: 'Arial'; 
                            color: #FAFAFA; background-color: #2D7308; text-align: center} </style> """, unsafe_allow_html = True)
                col11.markdown('<p class="message">Congratulations for achieving your target!</p>', unsafe_allow_html = True)



            # Title for radar chart
            st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
            st.markdown('<p class="subtitle">Your Standardized Marks Against Batch {} Median</p>'.format(pd.unique(filter_01['Batch'])), unsafe_allow_html = True)



            # Filters & preparing data for radar chart
            result_04 = result[['Semester', 'Session', 'Batch', 'Roll No.', 'Assignment', 'Midterm', 
                                'Online Participation', 'Group Project', 'FINALS']]

            result_04.rename(columns = {'FINALS' : 'Finals'}, inplace = True)
            result_04['Midterm'] = (result_04['Midterm'] / 20) * 10
            result_04['Group Project'] = (result_04['Group Project'] / 20) * 10
            result_04['Finals'] = (result_04['Finals'] / 40) * 10
            result_04 = result_04.round({"Midterm" : 1, "Group Project" : 1, "Finals" : 1})

            result_05 = result_04.groupby(['Batch', 'Semester', 'Session'], as_index = False)['Assignment', 'Midterm', 'Online Participation', 'Group Project', 'Finals'].median()

            filter_04 = result_04[(result_04['Semester'] == sem_option) & (result_04['Session'] == session_option) & (result_04['Roll No.'] == num_option)]
            filter_05 = result_05[(result_05['Semester'] == sem_option) & (result_05['Session'] == session_option)]


            # Radar chart on individual student performance & batch median
            categories = ['Assignment','Midterm','Online Participation', 'Group Project', 'Finals']
            fig2 = go.Figure()

            fig2.add_trace(go.Scatterpolar(r = [filter_05['Assignment'].iloc[0], filter_05['Midterm'].iloc[0], filter_05['Online Participation'].iloc[0], 
                            filter_05['Group Project'].iloc[0], filter_05['Finals'].iloc[0]], theta = categories, fill = 'toself', name = "Median", marker_color = '#E67517'))

            fig2.add_trace(go.Scatterpolar(r = [filter_04['Assignment'].iloc[0], filter_04['Midterm'].iloc[0], filter_04['Online Participation'].iloc[0], 
                            filter_04['Group Project'].iloc[0], filter_04['Finals'].iloc[0]], theta = categories, fill = 'toself', name = num_option, marker_color = '#3550F7'))

            fig2.update_layout(polar = dict(radialaxis = dict(visible = True, range = [0, 10]), bgcolor = "#1e2130"), showlegend = True)
            st.plotly_chart(fig2)

    
        else:
            image_denied()





    #### ------------ APP COMPONENT 3: GRADES PREDICTION (Predict Grades for Current New Batch of Students) ------------ ####
    elif choose_app == "Predict Grades":
        if student_main_option == 'NO':


            # Load results data and filter from Batch 6 till 9 only
            result = load_data_result()
            result_predict = result[result['Batch'].isin([6, 7, 8, 9])]
            result_predict = result_predict[['Assignment', 'Midterm', 'Online Participation', 'Alternative Assessment 01', 'TOTAL MARKS']]
        

        
            # Feature to upload results for current batch of students
            st.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
            st.markdown('<p class="caption">Please upload existing results here for your current batch of students..</p>', unsafe_allow_html = True)

            upload_expander = st.expander(label = 'Click "+" to open file uploader platform')
    
            with upload_expander:
                upload_file = st.file_uploader("", type = ['xlsx'], key = "2")
                st.write("\n")
                'NO DATA?? Click on this [link](https://docs.google.com/spreadsheets/d/1Z6bya63MMAE9NpE5qBiL1C6XoW5_uvFW//export?format=xlsx) to download an Excel dummy data for the current batch of students.'



            # Load data for results from new batch of students
            if upload_file is not None:
                new_result = pd.read_excel(upload_file)

                try:
                    # Deploy RF regression model
                    model, scaler = train_model(result_predict)
                    new_result2 = new_result[['Assignment', 'Midterm', 'Online Participation', 'Alternative Assessment 01']]
                    new_result2 = pd.DataFrame(scaler.fit_transform(new_result2))

                    # Obtain "Predicted Marks" & "Predicted GP & AA2" columns
                    Y_pred = model.predict(new_result2)
                    new_result['Predicted GP & AA2'] = Y_pred.tolist() - new_result.drop('Roll No.', axis = 1).sum(numeric_only = True, axis = 1)
                    new_result['Predicted Marks'] = Y_pred.tolist()
                    new_result = new_result.round(1)
                
                    # Obtain "Predicted Grade", "Action", and "Roll No." columns
                    new_result['Predicted Grade'] = new_result.apply(lambda x: grade_func(x['Predicted Marks']), axis = 1)
                    new_result['Action'] = new_result.apply(lambda x: action_func(x['Predicted Marks']), axis = 1)
                    new_result['Roll No.'] = np.arange(len(new_result)) + 1
                    new_result['Roll No.'] = "NEW" + new_result.apply(lambda x: number_func(x['Roll No.']), axis = 1)



                    # Title for bar chart
                    st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                        color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
                    st.markdown('<p class="subtitle">Predicted Grades for Current Batch of Students</p>', unsafe_allow_html = True)



                    # Bar chart on number of students for each predicted grade
                    new_result_01 = new_result.groupby('Predicted Grade', as_index = False).agg({"Action": "count"})
                    new_result_01.rename(columns = {'Action' : 'Frequency'}, inplace = True)

                    new_result_01['Seq Num'] = new_result_01.apply(lambda x: grade_seq_func(x['Predicted Grade']), axis = 1)
                    new_result_01.sort_values(by = 'Seq Num', inplace = True)


                    fig1 = go.Figure()

                    fig1.add_trace(go.Bar(name = "Excellent", x = new_result_01[new_result_01['Seq Num'].isin([1,2])]['Predicted Grade'], 
                                y = new_result_01[new_result_01['Seq Num'].isin([1,2])]['Frequency'], marker_color = '#0CA40C'))
                    
                    fig1.add_trace(go.Bar(name = "Average", x = new_result_01[new_result_01['Seq Num'].isin([3,4])]['Predicted Grade'], 
                                y = new_result_01[new_result_01['Seq Num'].isin([3,4])]['Frequency'], marker_color = '#D0B914'))

                    fig1.add_trace(go.Bar(name = "At-Risk", x = new_result_01[new_result_01['Seq Num'].isin([5,6,7,8,9,10,11])]['Predicted Grade'], 
                                y = new_result_01[new_result_01['Seq Num'].isin([5,6,7,8,9,10,11])]['Frequency'], marker_color = '#D03318'))
                    
                    fig1.update_layout(title = {'text': "Number of Students for each Predicted Grade", 'x': 0.5, 'xanchor': 'center'},
                                    xaxis_title_text = 'Predicted Grade', yaxis_title_text = 'No. of Students')
                    
                    
                    col1, col2, col3 = st.columns([1,12,1])
                    col1.write("")
                    col2.plotly_chart(fig1, use_column_width = True)
                    col3.write("")



                    # Recommendation based on chosen action
                    col4, col5 = st.columns([3,8])
                    col4.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                        color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
                    col4.markdown('<p class="caption">What actions do you plan to take??</p>', unsafe_allow_html = True)

                    action_option = col5.selectbox("", pd.unique(new_result['Action']), index = 2)
                    filter_01 = new_result[(new_result['Action'] == action_option)]

                    if action_option == "Engage (At-Risk)":
                        st.markdown(""" <style> .fonts {font-size: 16px ; font-family: 'Arial'; 
                                    color: #FAFAFA; background-color: #A81313; text-align: left} </style> """, unsafe_allow_html = True)
                        st.markdown('<p class="fonts">Huge effort is required! Please proceed to the list of past students under the "Past Results" tab and engage a few top students to be peer tutors.</p>', unsafe_allow_html = True)

                    elif action_option == "Explore (Average)":
                        st.markdown(""" <style> .fonts {font-size: 16px ; font-family: 'Arial'; 
                                    color: #FAFAFA; background-color: #8A6305; text-align: left} </style> """, unsafe_allow_html = True)
                        st.markdown('<p class="fonts">Effort is required! Do reach out to your students personally first. Only proceed to the list of past students under the "Past Results" tab and engage a few top students to be peer tutors, if deemeed necessary.</p>', unsafe_allow_html = True)

                    else:
                        st.markdown(""" <style> .fonts {font-size: 16px ; font-family: 'Arial'; 
                                    color: #FAFAFA; background-color: #2D7308; text-align: left} </style> """, unsafe_allow_html = True)
                        st.markdown('<p class="fonts">Minimal effort is required! Just encourage your students to complete their vital tasks on time.</p>', unsafe_allow_html = True)



                    # Caption for data table
                    st.write("\n")
                    st.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                        color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
                    st.markdown('<p class="caption">Choose the list of students that you wish to target..</p>', unsafe_allow_html = True)



                    # Data table based on filter by Action taken
                    fig2 = GridOptionsBuilder.from_dataframe(filter_01.drop(['Action'], axis =1))
                    fig2.configure_pagination(paginationAutoPageSize = True)
                    fig2.configure_side_bar() 
                    fig2.configure_selection('multiple', use_checkbox = True, groupSelectsChildren= "Group checkbox select children")
                    gridOptions = fig2.build()

                    grid_response = AgGrid(filter_01.drop(['Action'], axis =1), gridOptions = gridOptions, data_return_mode = 'AS_INPUT', 
                                        update_mode = 'MODEL_CHANGED', fit_columns_on_grid_load = False, theme = 'blue', 
                                        enable_enterprise_modules = True, height = 250, width = '100%', reload_data = False)


                    # Data table based on filter by Action taken (after selection)
                    df = grid_response['data']
                    selected = grid_response['selected_rows']
                    new_result_02 = pd.DataFrame(selected) 

                    try:
                        st.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                            color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
                        st.markdown('<p class="caption">Your confirmed list of students..</p>', unsafe_allow_html = True)

                        fig3 = st.dataframe(new_result_02[['Roll No.', 'Predicted Marks', 'Predicted Grade']].style.format({'Predicted Marks': '{:.1f}'}))

                    except:
                        st.error("No Data To Show!!")
                
                except:
                    st.error("No Data To Show!!")
            
            else:
                st.write('---') 

        else:
            image_denied()

    



    #### ------------ APP COMPONENT 4: SENTIMENT ANALYSIS (Analyse Feedback From Words of Closure) ------------ ####
    elif choose_app == "Sentiments":
        if student_main_option == 'NO':
            sentiment = load_data_sentiment()


            # Title for big number chart with expander
            st.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
            st.markdown('<p class="subtitle">Lifetime Number of Students Sentiments</p>', unsafe_allow_html = True)


            choice_expander = st.expander(label = 'Click "+" to choose your desired display of number')
            
            with choice_expander:
                col0A, col0B = st.columns([1,5])
                col0A.markdown(""" <style> .lifetime {font-size: 18px ; font-family: 'Arial'; 
                        color: #FAFAFA; text-align: left} </style> """, unsafe_allow_html = True)
                col0A.markdown('<p class="lifetime">View As:</p>', unsafe_allow_html = True)
                
                lifetime_option = col0B.radio("", ('Absolute Number', 'Percentage'))



            # Big number chart on lifetime numbers
            col1, col2, col3, col4 = st.columns([2,2,2,1])
            positive = sentiment['Sentiment'] == 'Positive'
            neutral = sentiment['Sentiment'] == 'Neutral'
            negative = sentiment['Sentiment'] == 'Negative'


            if lifetime_option == 'Absolute Number':
                col1.metric("Total Feedback", sentiment.shape[0])
                col2.metric("Positive", positive.sum())
                col3.metric("Neutral", neutral.sum())
                col4.metric("Negative", negative.sum())

            else:
                col1.metric("Total Feedback", "100%")
                col2.metric("Positive", "{}%".format(round(positive.sum() *100/sentiment.shape[0], 1)))
                col3.metric("Neutral", "{}%".format(round(neutral.sum() *100/sentiment.shape[0], 1)))
                col4.metric("Negative", "{}%".format(round(negative.sum() *100/sentiment.shape[0], 1)))



            # Filters by semester & session as well as preparing data for donut chart & histogram
            choice_expander2 = st.expander(label = 'Click "+" to choose your desired semester and session')

            with choice_expander2:
                col5, col6, col7 = st.columns([1,1,2])
                sem_option = col5.radio('Semester', pd.unique(sentiment['Semester']))
                session_option = col6.selectbox('Session', pd.unique(sentiment['Session']))
            

            filter_01 = sentiment[(sentiment['Semester'] == sem_option) & (sentiment['Session'] == session_option)]

            sentiment_01 = filter_01.groupby('Sentiment', as_index = False).agg({"Feedback": "count"})
            sentiment_01.rename(columns = {'Feedback' : 'Frequency'}, inplace = True)
            sentiment_01.sort_values(by = 'Sentiment', ascending = False, inplace = True)



            # Title for donut chart & histogram
            col8, col9 = st.columns(2)
            col8.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
            col8.markdown('<p class="subtitle">Sentiments</p>', unsafe_allow_html = True)

            col9.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
            col9.markdown('<p class="subtitle">Ratings Distribution</p>', unsafe_allow_html = True)



            # Donut chart on students' sentiments by batch
            fig1 = go.Figure(data = [go.Pie(labels = sentiment_01['Sentiment'], values = sentiment_01['Frequency'])])
            fig1.update_traces(hole = 0.7, hoverinfo = 'label+percent', textinfo = 'value', textfont_size = 20,
                            marker = dict(colors = ['green', 'yellow', 'red'], line = dict(color = '#000000', width = 2)))
            fig1.update_layout(title = {'text': "Semester {} Session {} - Batch {}".format(sem_option, session_option, pd.unique(filter_01['Batch'])), 'x': 0.5, 'xanchor': 'center'})
            
            col8.plotly_chart(fig1, use_container_width = True)



            # Histogram on ratings distribution by batch
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x = filter_01['Rating'], xbins = dict(start = 1, end = 5, size = 0.05), marker_color = '#EB89B5'))
            fig2.add_vline(x = filter_01['Rating'].median(), line_width = 3, line_color = "green", annotation_text = "Median", 
                        annotation_bgcolor = "blue", annotation_position = "left top")
            fig2.update_layout(title = {'text': "Semester {} Session {} - Batch {}".format(sem_option, session_option, pd.unique(filter_01['Batch'])),
                            'x': 0.5, 'xanchor': 'center'}, xaxis_title_text = 'Ratings', yaxis_title_text = 'No. of Feedbacks', bargap = 0.05, bargroupgap = 0.05)
            
            col9.plotly_chart(fig2, use_container_width = True)



            # Caption for sentiment exploration and option button
            col10, col11 = st.columns([2,3])
            col10.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                            color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
            col10.markdown('<p class="caption">Choose your desired sentiment..</p>', unsafe_allow_html = True)

            sent_option = col11.radio("", ('Positive', 'Neutral', 'Negative'))
            filter_02 = sentiment[(sentiment['Semester'] == sem_option) & (sentiment['Session'] == session_option) & (sentiment['Sentiment'] == sent_option)]
            


            # Title for word cloud & pictogram with expander
            wordcloud_rating_expander = st.expander(label = 'Click "+" to view the Wordcloud // Individual Feedback & Rating Visualizations')

            with wordcloud_rating_expander:
                col12, col13 = st.columns([1,1])
                col12.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                        color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
                col12.markdown('<p class="subtitle">Wordcloud: {}</p>'.format(sent_option), unsafe_allow_html = True)

                col13.markdown(""" <style> .subtitle {font-size: 22px ; font-family: 'Elephant'; 
                    color: #8F01A2; background-color: #F1E5E5; border-radius: 50%; text-align: center} </style> """, unsafe_allow_html = True)
                col13.markdown('<p class="subtitle">Feedback & Rating</p>', unsafe_allow_html = True)

                

                # Word cloud on sentiments by batch
                try:
                    fig3 = sc.gen_stylecloud(text = ' '.join(filter_02['Feedback 2']), icon_name = 'fas fa-apple-alt',
                                    palette='colorbrewer.diverging.Spectral_11', background_color='black', gradient='horizontal')
                    col12.image(Image.open('stylecloud.png'), use_column_width = True)

                except:
                    st.error("No Data To Show!!")
                    st.stop()


            
            # Caption for data table & rating slider
            col14, col15 = st.columns([3,2])
            col14.markdown(""" <style> .caption {font-size: 20px ; font-family: 'Arial'; 
                color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
            col14.markdown('<p class="caption">Choose your desired rating range & matric no.. Results would be populated at the top right..</p>', unsafe_allow_html = True)

            rating_slider = col15.slider("", min_value = float(filter_02['Rating'].min()) - 1, 
                                    max_value = float(filter_02['Rating'].max()), 
                                    value = (float(filter_02['Rating'].min() - 1), float(filter_02['Rating'].median())))
            
            

            # Data table on sentiment data
            filter_03 = sentiment[(sentiment['Semester'] == sem_option) & (sentiment['Session'] == session_option) & (sentiment['Sentiment'] == sent_option) & (sentiment['Rating'] >= rating_slider[0]) & (sentiment['Rating'] <= rating_slider[1])]
            
            fig4 = GridOptionsBuilder.from_dataframe(filter_03[['Matric No.', 'Feedback']])
            fig4.configure_pagination(paginationAutoPageSize = True)
            fig4.configure_side_bar() 
            fig4.configure_selection('single', use_checkbox = True, groupSelectsChildren= "Group checkbox select children")
            gridOptions = fig4.build()

            grid_response = AgGrid(filter_03[['Matric No.', 'Feedback', 'Rating']], gridOptions = gridOptions, data_return_mode = 'AS_INPUT', 
                                update_mode = 'MODEL_CHANGED', fit_columns_on_grid_load = True, theme = 'blue', 
                                enable_enterprise_modules = True, height = 250, width = '100%', reload_data = True)

        

            # Dataframe on sentiment data (after selection)
            df = grid_response['data']
            selected = grid_response['selected_rows']
            sentiment_02 = pd.DataFrame(selected)
            

            with wordcloud_rating_expander:
                # Captions & display of matric no. & dataframe for feedback
                try:
                    col13.markdown(""" <style> .matric {font-size: 24px ; font-family: 'Arial Black'; 
                        color: #0E1117; background-color: #FF4B4B; border-radius: 5%; text-align: center} </style> """, unsafe_allow_html = True)
                    col13.markdown('<p class="matric">Matric No: {}</p>'.format(sentiment_02['Matric No.'].iloc[0]), unsafe_allow_html = True)
                    
                    col13.markdown(""" <style> .feedback {font-size: 17px ; font-family: 'Arial'; 
                        color: #F19A2B; text-align: left} </style> """, unsafe_allow_html = True)
                    col13.markdown('<p class="feedback">Hover over to read full feedback..</p>', unsafe_allow_html = True)

                    fig5 = col13.dataframe(sentiment_02[['Feedback']])



                    # Pictogram on star rating
                    fig6 = rating_func(sentiment_02['Rating'].iloc[0])
                    col13.image(fig6, use_column_width = True)
                    
                    col13.markdown(""" <style> .rating {font-size: 35px ; font-family: 'Arial Black'; 
                        color: #EAC528; text-align: center} </style> """, unsafe_allow_html = True)
                    col13.markdown('<p class="rating">Rating: {}</p>'.format(sentiment_02['Rating'].iloc[0]), unsafe_allow_html = True)


                except:
                    col13.error("No Data To Show!!")
                    st.stop()
            

        else:
            image_denied()







##### ------------------ DASHBOARD COMPONENT 2: USER GUIDE ------------------ #####
elif choose_dashboard == "User Guide":
    st.write("\n")
    'Click on this [link](https://drive.google.com/file/d/1_hjDKvwrCWLlVgBI2lgn-HZGPvOaE38H/view?usp=sharing) to access the backup User Guide if document fails to display here.'
    show_pdf('user_guide.pdf')




##### ------------------ DASHBOARD COMPONENT 3: SOURCE CODE ------------------ #####
elif choose_dashboard == "Source Code":
    st.write("\n")
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        'Click on this [link](https://github.com/aniqkaw/JomCapai_Dashboard) to access the files in Github.'
        image_github()

    with col3:
        st.write("")




##### ------------------ DASHBOARD COMPONENT 4: ABOUT AUTHOR ------------------ #####
elif choose_dashboard == "About Author":
    st.write("\n")
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        'Click on this [link](https://www.linkedin.com/in/aniqkaw/) to access his LinkedIn Profile.'
        image_author()

    with col3:
        st.write("")
