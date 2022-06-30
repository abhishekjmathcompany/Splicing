
from catboost import Pool, cv, CatBoostClassifier
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random



def predict_quality(model, df):
    
    predictions_data = model.predict(df)
    probability = model.predict_proba(df)
    if predictions_data[0] == 1:
        return "Good Splice" +'\n'+" Probability: " + str(round(probability[0][1],3))
    if predictions_data[0] == 0:
        return "Bad Splice" +'\n'+" Probability: " + str(round(probability[0][1],3))
    return 0
    
model = CatBoostClassifier().load_model('cb_clf.pkl', format='cbm')

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center; color: black;'>Optimal Splice Machine settings Prediction</h1>", unsafe_allow_html=True)
st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


image = Image.open("logo.png") 

st.sidebar.image(image, caption='Solutions for your Journey') 


col1, col3, col2 = st.columns([5,0.5,1])

col1.write('***** This application is just a POC, features selected are not final, more work needs to be done to find out the best ones')




st.sidebar.title('Adjust Tweakable parameters')


image = Image.open('./Shap.png')
col1.image(image, caption='SHAP Feature imporatnce')
col1.write('''Note: while SHAP shows the contribution or the importance of each feature on the prediction
 of the model, it does not evaluate the quality of the prediction itself.'''    )

imp_features = ['PM_1PAP_WAIT_OPERATOR',
 'PM_BPSC_BUSY',
 'PM_BPSC_TOTAL',
 'PM_1PAP_TOTAL',
 'PM_1PAP_SAFETY',
 '1BPR_CYCLE_TRANSPORT',
 '2BAP_SPLICE_MAX_OFF_CENTER',
 '2BAP_BODY_MIN_OFF_CENTER',
 'BPPR_BP1_CYCLE_CUTTING',
 '2BAP_SPLICE_MIN_OFF_CENTER']
 
min_list=[0.0, 0.087997437, 7.00, 5.103999615, 0.0, 0.0, -1.148609161, -4.898849487, 7.945000172, -5.035800934]
max_list = [383.4790039, 83.45098877, 194.4089966, 1002.635986, 226.4299927, 239.4589996, 2.882957458, 2.370239258, 27.74399948, 2.267749786]       
median_list = [370.00, 65.50, 180.00, 5.10, 10.0, 11.30500031, -0.77, 0.07065773, 8.730999947, 0.133998871]
 
target = ['CCMO_BP1_LFT_SPLICE_LENGTH']



slider_list=[]
for i,j,k,l in zip(imp_features,min_list,max_list,median_list):
    slider_list.append(st.sidebar.slider(label = i,key=i,min_value = j,
                       max_value =  k,
                         value =  l,
                       step = 1.0))
    

features_df = pd.DataFrame([slider_list],columns=imp_features)
features_df_new = features_df.T
features_df_new.reset_index(inplace=True)
#print(type(features_df_new))
features_df_new.columns = ['Features','Values']

#################RANDOMized#############################

#def _update_slider(key,value_list):
 #   st.session_state["test_slider"] = value
    
if col2.button(label = "Predict best settings", help = "Click on this button to find the best parameter settings for the machine"):
    random_df = pd.DataFrame(columns=imp_features)
    for i in range(1,500):
    
        f1 = round(random.uniform(0.0, 383.4),2)
        f2 = round(random.uniform(0.08, 83.45),2)
        f3 = round(random.uniform(1.5, 194.4),2)
        f4 = round(random.uniform(5.1, 1002.63),2)
        f5 = round(random.uniform(0.0, 226.4),2)
        f6 = round(random.uniform(0.0, 239.45),2)
        f7 = round(random.uniform(-1.14, 2.88),2)
        f8 = round(random.uniform(-4.89, 2.37),2)
        f9 = round(random.uniform(7.94, 27.74),2)
        f10 = round(random.uniform(-5.03, 2.26),2)
    
        input_list_1 = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
        random_df.loc[len(random_df)] = input_list_1
        
    #prediction_1 = model.predict_proba(input_list_1)
    proba_df = pd.DataFrame(model.predict_proba(random_df),columns=['proba_0','proba_1'])
    final_rand_df = pd.concat([random_df, proba_df], axis=1)
    final_rand_df = final_rand_df.sort_values(by='proba_1',ascending=False)

    randomized_list = final_rand_df.iloc[:1,:-2].T.reset_index().iloc[:,-1:]
    randomized_list = list(randomized_list.iloc[:,-1])
    #randomized_list = list(final_rand_df.iloc[:1,:-2])
    
    #print("=======")
    #print(len(imp_features))
    #print("+++++++++++++")
    #print(len(randomized_list))
    
    
    #for i,j,k,l in zip(imp_features,min_list,max_list,median_list):
        #if "test_slider" not in st.session_state:
            #st.session_state[i] = 0
        #st.sidebar.slider(label = i,key = i,min_value = j,
                       # max_value =  k,
                       #     value =  l,
                       # step = 1.0)
    
  
    features_df = pd.DataFrame(columns=imp_features)  
    features_df.loc[len(randomized_list)] = randomized_list
    
    features_df_new = features_df.T
    features_df_new.reset_index(inplace=True)
    #   print(type(features_df_new))
    features_df_new.columns = ['Features','Values']

col1.table(features_df_new)  


####################RANDOMized##########################    
    
  
prediction = predict_quality(model, features_df)
    
col2.write('## Prediction:')
col2.title(prediction)

col2.download_button(
     label="Download as CSV",
     data=features_df_new.to_csv(index=False),
     file_name='Best settings.csv',
     mime='text/csv',
 )
