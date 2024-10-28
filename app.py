# app.py
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

class ChiliRecommender:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.model = NearestNeighbors(n_neighbors=3, metric='euclidean')
        
    def preprocess_data(self, df):
        # Create copy of dataframe
        processed_df = df.copy()
        
        # Convert categorical variables to numerical
        categorical_columns = ['Tipe', 'Kebutuhan Air', 
                             'Kebutuhan Sinar Matahari', 'Kondisi Tanam']  # Removed disease resistance
        
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
        
        # Handle altitude range
        processed_df['Altitude_Min'] = processed_df['Altitude (m)'].apply(lambda x: float(str(x).split('-')[0]))
        processed_df['Altitude_Max'] = processed_df['Altitude (m)'].apply(lambda x: float(str(x).split('-')[1]))
        processed_df['Altitude_Avg'] = (processed_df['Altitude_Min'] + processed_df['Altitude_Max']) / 2
        
        # Handle temperature range
        processed_df['Temp_Min'] = processed_df['Toleransi Suhu (°C)'].apply(lambda x: float(str(x).split('-')[0]))
        processed_df['Temp_Max'] = processed_df['Toleransi Suhu (°C)'].apply(lambda x: float(str(x).split('-')[1]))
        processed_df['Temp_Avg'] = (processed_df['Temp_Min'] + processed_df['Temp_Max']) / 2
        
        # Handle productivity range
        processed_df['Productivity_Min'] = processed_df['Produktivitas (Ton/Hektar)'].apply(lambda x: float(str(x).split('-')[0]))
        processed_df['Productivity_Max'] = processed_df['Produktivitas (Ton/Hektar)'].apply(lambda x: float(str(x).split('-')[1]))
        processed_df['Productivity_Avg'] = (processed_df['Productivity_Min'] + processed_df['Productivity_Max']) / 2
        
        # Select features for modeling
        self.features = ['Altitude_Avg', 'Temp_Avg'] + categorical_columns
        
        # Scale the features
        self.feature_matrix = self.scaler.fit_transform(processed_df[self.features])
        self.processed_df = processed_df
        
        return self.feature_matrix
    
    def transform_user_input(self, altitude, temperature, water_need, sunlight_need, land_condition):  # Removed disease_resistance parameter
        # Create a dictionary with user input
        user_data = {
            'Altitude_Avg': altitude,
            'Temp_Avg': temperature,
            'Kebutuhan Air': water_need,
            'Kebutuhan Sinar Matahari': sunlight_need,
            'Kondisi Tanam': land_condition,
            'Tipe': 'Hibrida'  # Default value, can be modified based on preference
        }
        
        # Transform categorical variables
        for col in ['Kebutuhan Air', 'Kebutuhan Sinar Matahari', 'Kondisi Tanam', 'Tipe']:  # Removed disease resistance
            if col in user_data:
                user_data[col] = self.label_encoders[col].transform([user_data[col]])[0]
        
        # Create feature vector
        user_vector = np.array([user_data[feature] for feature in self.features])
        
        # Scale the vector
        user_vector_scaled = self.scaler.transform(user_vector.reshape(1, -1))
        
        return user_vector_scaled

    def train(self, feature_matrix):
        self.model.fit(feature_matrix)
    
    def get_recommendations(self, user_input_vector):
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(user_input_vector)
        
        # Get recommendations
        recommendations = []
        for idx in indices[0]:
            variety = self.processed_df.iloc[idx]
            recommendations.append({
                'Varietas': variety['Varietas'],
                'Tipe': variety['Tipe'],
                'Produktivitas': f"{variety['Productivity_Min']}-{variety['Productivity_Max']} Ton/Hektar",
                'Ketahanan': variety['Ketahanan Terhadap Penyakit'],  # Keep this in output for information
                'Similarity_Score': round((1 - distances[0][list(indices[0]).index(idx)]) * 100, 2)
            })
        
        return recommendations

# Initialize the recommender
recommender = ChiliRecommender()
# Load and preprocess data
df = pd.read_csv('Data Varietas Tanaman Cabai - Sheet1.csv')
feature_matrix = recommender.preprocess_data(df)
recommender.train(feature_matrix)

class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
            
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get user input vector
        user_vector = recommender.transform_user_input(
            float(data['altitude']),
            float(data['temperature']),
            data['water_need'],
            data['sunlight_need'],
            data['land_condition']
        )
        
        # Get recommendations
        recommendations = recommender.get_recommendations(user_vector)
            
         # Create response data
        response_data = {
            'success': True,
            'recommendations': recommendations
        }
        
        # Convert to JSON using custom encoder
        return Response(
            json.dumps(response_data, cls=NumpyEncoder),
            mimetype='application/json'
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)