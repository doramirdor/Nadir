import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.complextiy.analyzer import ComplexityAnalyzer

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic text complexity data
    """
    texts = []
    complexities = []

    # Generate diverse text samples
    for _ in range(num_samples):
        # Vary text length and linguistic complexity
        length = np.random.randint(10, 500)
        text = ' '.join(['word'] * length)
        
        # Simulate linguistic features
        if np.random.random() < 0.3:
            text += ' '.join(['complex-term'] * np.random.randint(1, 10))
        
        texts.append(text)

    return texts

def train_complexity_model():
    """
    Train a machine learning model to predict text complexity
    """
    # Generate synthetic training data
    texts = generate_synthetic_data()
    
    # Use existing complexity analyzer as ground truth
    analyzer = ComplexityAnalyzer()
    
    # Calculate complexity for each text
    complexities = [analyzer.calculate_complexity(text) for text in texts]
    
    features = []
    # Feature extraction (basic example)
    for text in texts:
        features.append([
            len(text.split()),  # Word count
            len(set(text.split())),  # Unique word count
            text.count(' ') / len(text) if len(text) > 0 else 0  # Spacing density
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, complexities, test_size=0.2
    )
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate model
    score = model.score(X_test, y_test)
    print(f"Model R² Score: {score}")
    
    return model

if __name__ == "__main__":
    train_complexity_model()