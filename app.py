from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Allowed file types
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create the model dynamically based on user input
def create_optimized_model(input_shape, neurons, layers, learning_rate, dropout_rate, optimizer_type, problem_type):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_shape, activation='relu'))
    
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    if problem_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))  # For binary classification
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(Dense(1, activation='linear'))  # For regression
        loss = 'mse'
        metrics = ['mae']
    
    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

# Function to train the model and plot results
def train_model(X_train, y_train, X_val, y_val, neurons, layers, learning_rate, dropout_rate, optimizer_type, batch_size, epochs, problem_type):
    input_shape = X_train.shape[1]
    model = create_optimized_model(input_shape, neurons, layers, learning_rate, dropout_rate, optimizer_type, problem_type)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr], verbose=1)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if problem_type == 'classification':
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.ylabel('Accuracy')
    else:
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.ylabel('MAE')
    
    plt.title('Performance Over Epochs')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.savefig('static/training_history.png')  # Save the plot in the static folder for display

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load the dataset
            df = pd.read_csv(filepath)
            
            # Feature selection based on columns
            X = df.drop('gender', axis=1)  # Assuming 'gender' is the label
            y = df['gender'].map({'Male': 1, 'Female': 0})  # Mapping binary labels

            # Split data
            test_size = float(request.form.get('test_size', 0.2))
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

            # Standard scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Get hyperparameters
            neurons = int(request.form.get('neurons', 128))
            layers = int(request.form.get('layers', 2))
            learning_rate = float(request.form.get('learning_rate', 0.001))
            dropout_rate = float(request.form.get('dropout_rate', 0.3))
            optimizer_type = request.form.get('optimizer', 'adam')
            batch_size = int(request.form.get('batch_size', 32))
            epochs = int(request.form.get('epochs', 10))
            problem_type = request.form.get('problem_type', 'classification')

            # Train the model and save the plot
            train_model(X_train, y_train, X_val, y_val, neurons, layers, learning_rate, dropout_rate, optimizer_type, batch_size, epochs, problem_type)

            return render_template('result.html', plot_url='static/training_history.png')

    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = 'supersecretkey'
    app.run(debug=True)
