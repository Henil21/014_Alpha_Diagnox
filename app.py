from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib
import cv2
import os
import pathlib
from PIL import Image
import google.generativeai as google_genai
import io
import markdown  # Import markdown module for rendering Markdown
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from array import array
from PIL import Image
import pickle

app = Flask(__name__)

# TensorFlow Model Prediction
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Google GenAI system prompts for Nimbus
system_prompts = """You are Nimbus, a domain expert in medical analysis. You are tasked with examining user queries regarding health. 
Your expertise will help in identifying or discovering any anomalies, diseases, conditions or any health issues that might be present from the provided description or symptoms.

Your key responsibilities:
1. Detailed Analysis : Provide a detailed response, ask for additional information if needed.
2. Analysis Report : Document all the findings and clearly articulate them in a structured format.
3. Recommendations : Basis the analysis, suggest remedies, tests or treatments as applicable. 
4. Treatments : If applicable, lay out detailed treatments which can help in faster recovery. Also provide a diet plan based on the condition.

Important Notes to remember:
1. Scope of response : Only respond if the query pertains to human health issues, or medical related. If it is not medical related just tell the user to only pass medical related images and or queries.
2. Clarity of query : In case the query is unclear ask for additional information
3. Disclaimer : Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."
4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above.

Please provide the final response with these 5 headings: 
Detailed Analysis, Analysis Report, Recommendations, Treatments, and Diet Plan."""

# Route for home page
@app.route('/')
def ho_me():
    return render_template('home.html')
@app.route('/home')
def home():
    return render_template('home.html')

# Route for brain-related model prediction
# @app.route('/brain', methods=['GET', 'POST'])
# def brain():
#     if request.method == 'POST':
#         uploaded_file = request.files.get('file')  # Get the uploaded file
#         if uploaded_file:
#             # Save the uploaded file temporarily for processing
#             filepath = f"static/uploads/{uploaded_file.filename}"
#             uploaded_file.save(filepath)

#             # Load and preprocess the image
#             try:
#                 # Load the image with OpenCV or Pillow
#                 img = cv2.imread(filepath)  # Reads in BGR format
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#                 img = cv2.resize(img, (224, 224))  # Resize to model's input size
#                 img = np.expand_dims(img, axis=0)  # Add batch dimension
#                 img = img / 255.0  # Normalize pixel values

#                 # Load the TensorFlow model
#                 model = tf.keras.models.load_model("models/brain_model.h5")

#                 # Make prediction
#                 predictions = model.predict(img)
#                 predicted_class = np.argmax(predictions)  # Assuming classification task

#                 # Return the prediction
#                 return f"Prediction: {predicted_class} (Confidence: {np.max(predictions) * 100:.2f}%)"
#             except Exception as e:
#                 return f"Error processing the image: {str(e)}"
#         else:
#             return "No file uploaded."
#     return render_template('brain.html')



model = tf.keras.models.load_model("models/brain_model.h5")

# Define the class names
data_dir = pathlib.Path("Tumor_MRI/brain_tumor_dataset/")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

# Function to preprocess image
def load_prep_img(file, img_shape=224):
    """
    Load and preprocess image from file
    """
    # Read image file
    img = tf.io.read_file(file)
    # Decode image
    img = tf.image.decode_image(img)
    # Resize image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Normalize image
    img = img / 255.0
    return img
def pred_and_plot(filename,class_name=class_names):
        """
        Imports an image locate at filename ,make a prediction with model
        and plot the images with the predicted class as title
        """
      #  import the target image and preprocess it
        img=load_prep_img(filename)
        pred=model.predict(tf.expand_dims(img,axis=0))

@app.route('/brain', methods=['GET', 'POST'])
def brain():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')  # Get the uploaded file
        if uploaded_file:
            # Save the uploaded file temporarily for processing
            filepath = f"static/uploads/{uploaded_file.filename}"
            uploaded_file.save(filepath)

            # Load and preprocess the image
            try:
                # Load the image with OpenCV or Pillow
                img = cv2.imread(filepath)  # Reads in BGR format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = cv2.resize(img, (224, 224))  # Resize to model's input size
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                img = img / 255.0  # Normalize pixel values

                # Load the TensorFlow model
                model = tf.keras.models.load_model("models/brain_model.h5")

                # Make prediction
                predictions = model.predict(img)
                predicted_class = np.argmax(predictions) 
                 # Assuming classification task
                confidence=np.max(predictions) * 100
                if confidence>=70:
                    pred=1 
                else:
                    pred=0

                # Return the prediction
                return f"Prediction: {pred} (Confidence: {np.max(predictions) * 100:.2f}%)"
            except Exception as e:
                return f"Error processing the image: {str(e)}"
        else:
            return "No file uploaded."
    return render_template('brain.html')

medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))


@app.route('/recommend', methods=['POST', 'GET'])
def recommend_medicines():
    # Get the selected medicine name from the form
    selected_medicine_name = request.form.get('medicine_name')
    
    # Get recommendations
    recommendations = recommend(selected_medicine_name)
    
    # Check if the result is an error message
    if isinstance(recommendations, str):  # Error message case
        return render_template('index.html', 
                               medicines=medicines['Drug_Name'].values, 
                               error=recommendations, 
                               selected_medicine=selected_medicine_name)
    
    # Render recommendations on the page
    return render_template('index.html', 
                           medicines=medicines['Drug_Name'].values, 
                           recommendations=recommendations, 
                           selected_medicine=selected_medicine_name)

def recommend(medicine):
    # Check if the medicine exists in the DataFrame
    if medicine not in medicines['Drug_Name'].values:
        return f"Medicine '{medicine}' not found. Please select a valid medicine."

    # Proceed with recommendation
    try:
        medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
        distances = similarity[medicine_index]
        medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_medicines = [medicines.iloc[i[0]].Drug_Name for i in medicines_list]
        return recommended_medicines
    except Exception as e:
        return f"An error occurred while processing the recommendation: {str(e)}"




@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Extract input values from the form
        Age = int(request.form['Age'])
        CP = int(request.form['CP'])
        trestbps = int(request.form['trestbps'])
        Cholesterol = int(request.form['Cholesterol'])
        thalachh = int(request.form['thalachh'])
        oldpeak = float(request.form['oldpeak'])

        # Create DataFrame for prediction
        data = {'Age': [Age], 'CP': [CP], 'trestbps': [trestbps], 'Cholesterol': [Cholesterol],
                'thalachh': [thalachh], 'oldpeak': [oldpeak]}
        df = pd.DataFrame(data)

        # Predict with the model
        model_heart=tf.keras.models.load_model("models/Heart_attack.h5")
        prediction = model_heart.predict(df)
        result = tf.math.round(prediction)[0][0]

        # Return result
        if result == 1:
            return render_template('heart.html', prediction_text='Positive! Seek expert advice.',
                                   warning='This AI model has accuracy of 80 percent. The developer takes no liability for any false prediction.')
        else:
            return render_template('heart.html', prediction_text='Negative! Seek expert advice.',
                                   warning='This AI model has accuracy of 80 percent. The developer takes no liability for any false prediction.')
    except Exception as e:
        return render_template('heart.html', prediction_text='Error occurred. Please check your inputs.', error=str(e))

    

@app.route('/vision_assist', methods=['POST', 'GET'])
def vision_assist():
    if request.method == 'POST':
        text = request.form.get("text", "")
        uploaded_file = request.files.get("image")

        if uploaded_file:
            try:
                # Read image content into memory
                image_data = uploaded_file.read()
                img = Image.open(io.BytesIO(image_data))

                # Prepare input for the model (make sure your model expects the image in this format)
                input_text = f"Analyze the following health-related query and image: {text}"

                
                model = google_genai.GenerativeModel(model_name = 'gemini-1.5-flash', system_instruction = system_prompts)
                google_genai.configure(api_key='AIzaSyD0FHc4z5c8HqDpfDvDUcRi4MNLOyoy-_E')  # Replace with actual key
                generation_config = {
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 0,
                    "max_output_tokens": 8192,
                    }
                response = model.generate_content([text, img])
                chatbot_response = response.text if response.text else "Currently cannot generate, please wait! Try again later!"
              
                # Convert the response to markdown format for rendering
                chatbot_response_markdown = markdown.markdown(chatbot_response)

                # Pass the response to the new template
                return render_template('response.html', response=chatbot_response_markdown)

            except Exception as e:
                return f"Model generation failed! {str(e)}"
        else:
            return "No image uploaded."
    else:
        return render_template('vision_assist.html')
    

@app.route('/contect', methods=['GET', 'POST'])
def contect():
     return render_template('contect.html')


@app.route('/diabetes', methods=['GET', 'POST'])


def diabetes():
    
    Data=pd.read_csv('diabetes (1).csv')

    Data.head()
    data=Data.drop('Outcome',axis=1)
    outc=Data["Outcome"]

    ct=make_column_transformer(
                  (MinMaxScaler(),['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']),
              )
    outc=Data["Outcome"]

    X=Data.drop("Outcome",axis=1)

    y=Data["Outcome"]
    X_train,X_test,Y_train,Y_test=train_test_split(data,outc,test_size=0.2,random_state=42)
    ct.fit(X_train)

    if request.method == 'POST':
        # Extract user input from the form
        pregnancies = int(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })

        # Load the trained model
        loading2 = tf.keras.models.load_model("models/diabetes.h5")
        input_scaled = ct.transform(input_data)

        # Make the prediction
        prediction = loading2.predict(input_scaled)

        # Interpret the prediction
        result = 'Positive' if tf.math.round(prediction) == 1 else 'Negative'

        # Return the result to the template
        return render_template('diabetes.html', 
                               result=result, 
                               confidence=float(prediction[0][0]) * 100)

    # If the method is GET, show the form without predictions
    return render_template('diabetes.html')

@app.route('/help', methods=['GET', 'POST'])
def help():
     return render_template('help.html')


if __name__ == "__main__":
    app.run(debug=True)  # Fixed comment style
