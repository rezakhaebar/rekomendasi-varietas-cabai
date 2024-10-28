import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

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
    def calculate_similarity_score(self, distance):
        """
        Convert Euclidean distance to similarity score (0-100%)
        """
        # Menggunakan fungsi eksponensial untuk mengkonversi jarak ke similarity
        # Semakin kecil jarak, semakin tinggi similarity
        similarity = np.exp(-distance)
        # Mengkonversi ke persentase
        similarity_percentage = similarity * 100
        return similarity_percentage
    
    def get_recommendations(self, user_input_vector):
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(user_input_vector)
        
        # Get recommendations
        recommendations = []
        for idx in enumerate(indices[0]):
            similarity_score = self.calculate_similarity_score(distances[0][i])
            variety = self.processed_df.iloc[idx]
            recommendations.append({
                'Varietas': variety['Varietas'],
                'Tipe': variety['Tipe'],
                'Produktivitas': f"{variety['Productivity_Min']}-{variety['Productivity_Max']} Ton/Hektar",
                'Ketahanan': variety['Ketahanan Terhadap Penyakit'],  # Keep this in output for information
                'Similarity_Score': round((1 - distances[0][list(indices[0]).index(idx)]) * 100, 2)
            })
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Data Varietas Tanaman Cabai - Sheet1.csv')
    
    # Initialize recommender
    recommender = ChiliRecommender()
    
    # Preprocess data and train model
    feature_matrix = recommender.preprocess_data(df)
    recommender.train(feature_matrix)
    
    # Example user input
    user_input = {
        'altitude': 500,  # meters
        'temperature': 28,  # Celsius
        'water_need': 'Sedang',  # Tinggi/Sedang/Rendah
        'sunlight_need': 'Tinggi',  # Tinggi/Sedang/Rendah
        'land_condition': 'Dataran menengah'  # Dataran rendah/menengah/tinggi
    }
    
    # Get user input vector
    user_vector = recommender.transform_user_input(
        user_input['altitude'],
        user_input['temperature'],
        user_input['water_need'],
        user_input['sunlight_need'],
        user_input['land_condition']
    )
    
    # Get recommendations
    recommendations = recommender.get_recommendations(user_vector)
    
    # Print recommendations
    print("\nTop Chili Variety Recommendations:")
    print("-" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Variety: {rec['Varietas']}")
        print(f"   Type: {rec['Tipe']}")
        print(f"   Productivity: {rec['Produktivitas']}")
        print(f"   Disease Resistance: {rec['Ketahanan']}")  # Still showing this in output for information
        print(f"   Similarity Score: {rec['Similarity_Score']}%")