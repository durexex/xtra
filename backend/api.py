import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import missingno as msno
import seaborn as sns

# Use a non-interactive backend for matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)

# Store dataframe in memory
df = None

def _build_dataframe_info(target_df: pd.DataFrame) -> dict:
    """
    Assemble a structured payload similar to pandas.DataFrame.info().
    This avoids string parsing on the frontend while keeping the info concise.
    """
    buffer = io.StringIO()
    target_df.info(buf=buffer)
    info_text = buffer.getvalue()
    buffer.close()

    non_null_counts = target_df.notnull().sum().to_dict()
    dtypes = {col: str(dtype) for col, dtype in target_df.dtypes.items()}
    memory_usage_bytes = target_df.memory_usage(deep=True).sum()

    return {
        'rows': int(target_df.shape[0]),
        'columns': int(target_df.shape[1]),
        'index': str(target_df.index),
        'nonNull': non_null_counts,
        'dtypes': dtypes,
        'memoryUsageBytes': int(memory_usage_bytes),
        'infoText': info_text
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            return jsonify({'message': 'File uploaded successfully', 'columns': list(df.columns)}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/head', methods=['GET'])
def get_head():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    return jsonify(df.head(100).to_dict(orient='records')), 200

@app.route('/dataframe-info', methods=['GET'])
def get_dataframe_info():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    try:
        metadata = _build_dataframe_info(df)
        return jsonify(metadata), 200
    except Exception as e:
        # TODO: add structured logging once logging stack is defined.
        return jsonify({'error': f'Failed to collect dataframe info: {str(e)}'}), 500

@app.route('/describe', methods=['GET'])
def get_describe():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    return jsonify(df.describe().to_dict()), 200

@app.route('/null-values', methods=['GET'])
def get_null_values():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    missing_values = ["Missing", "missing", "MISSING", "?"]
    df_temp = df.replace(missing_values, pd.NA)
        
    null_counts = df_temp.isnull().sum()
    
    total = null_counts.to_dict()

    return jsonify(total), 200

@app.route('/groupby', methods=['GET'])
def get_groupby():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    column = request.args.get('column')
    if not column or column not in df.columns:
        return jsonify({'error': 'Invalid or missing column parameter'}), 400
        
    try:
        grouped_describe = df.groupby(column).describe()
        # Convert tuple columns to string to avoid JSON error
        grouped_describe.columns = ['_'.join(col).strip() for col in grouped_describe.columns.values]
        return jsonify(grouped_describe.to_dict(orient='index')), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scatterplot', methods=['GET'])
def get_scatterplot():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    x_col = request.args.get('x_col')
    y_col = request.args.get('y_col')

    if not x_col or x_col not in df.columns:
        return jsonify({'error': 'Invalid or missing x_col parameter'}), 400
    if not y_col or y_col not in df.columns:
        return jsonify({'error': 'Invalid or missing y_col parameter'}), 400
        
    try:
        plt.figure()
        plt.scatter(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        
        # Save plot to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode image to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return jsonify({'image': image_base64}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/boxplot', methods=['GET'])
def get_boxplot():
   if df is None:
       return jsonify({'error': 'No dataframe loaded'}), 400
   
   column = request.args.get('column')
   if not column or column not in df.columns:
       return jsonify({'error': 'Invalid or missing column parameter'}), 400
       
   try:
       plt.figure()
       sns.boxplot(y=df[column])
       plt.title(f'Boxplot of {column}')
       
       buf = io.BytesIO()
       plt.savefig(buf, format='png')
       buf.seek(0)
       
       image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
       buf.close()
       
       return jsonify({'image': image_base64}), 200
       
   except Exception as e:
       return jsonify({'error': str(e)}), 500

@app.route('/histogram', methods=['GET'])
def get_histogram():
  if df is None:
      return jsonify({'error': 'No dataframe loaded'}), 400
  
  column = request.args.get('column')
  if not column or column not in df.columns:
      return jsonify({'error': 'Invalid or missing column parameter'}), 400
      
  try:
      plt.figure()
      sns.histplot(df[column], kde=True)
      plt.title(f'Histogram of {column}')
      
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      buf.seek(0)
      
      image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
      buf.close()
      
      return jsonify({'image': image_base64}), 200
      
  except Exception as e:
      return jsonify({'error': str(e)}), 500

@app.route('/scatterplot3d', methods=['GET'])
def get_scatterplot3d():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    x_col = request.args.get('x_col')
    y_col = request.args.get('y_col')
    z_col = request.args.get('z_col')

    if not x_col or x_col not in df.columns:
        return jsonify({'error': 'Invalid or missing x_col parameter'}), 400
    if not y_col or y_col not in df.columns:
        return jsonify({'error': 'Invalid or missing y_col parameter'}), 400
    if not z_col or z_col not in df.columns:
        return jsonify({'error': 'Invalid or missing z_col parameter'}), 400
        
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[x_col], df[y_col], df[z_col])
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        plt.title(f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}')
        
        # Save plot to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode image to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return jsonify({'image': image_base64}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/traintest', methods=['GET'])
def get_traintest():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    n_neighbors = request.args.get('n_neighbors', default=3, type=int)

    x = df[["mean_radius", "mean_area", "mean_perimeter", "mean_texture", "mean_smoothness"]]
    y = df["diagnosis"]
    
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size = 0.2,
                                                        stratify = y,
                                                        random_state = 42)
    
    modelo_classificador = KNeighborsClassifier(n_neighbors=n_neighbors)
    modelo_classificador.fit(x_train, y_train)
    
    
    y_pred = modelo_classificador.predict(x_test)
    acuracia = accuracy_score(y_test, y_pred)

    train_size = len(x_train)
    test_size = len(x_test)
    
    diagnosis_counts = df['diagnosis'].value_counts()
    diag_0_count = diagnosis_counts.get(0, 0)
    diag_1_count = diagnosis_counts.get(1, 0)

    html_response = f"""
    <div>
        <h3>Resultados do Treinamento e Teste</h3>
        <p><strong>Tamanho do Conjunto de Treino:</strong> {train_size}</p>
        <p><strong>Tamanho do Conjunto de Teste:</strong> {test_size}</p>
        <h4>Contagem de Diagnósticos no Dataset Completo</h4>
        <p><strong>Diagnóstico 0:</strong> {diag_0_count}</p>
        <p><strong>Diagnóstico 1:</strong> {diag_1_count}</p>
        <h4>Performance do Modelo</h4>
        <p><strong>Acurácia:</strong> {acuracia:.4f}</p>
    </div>
    """
    
    return jsonify({'html': html_response}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    data = request.get_json()
    if not data or not all(k in data for k in ['mean_radius', 'mean_area', 'mean_perimeter', 'mean_texture', 'mean_smoothness', 'n_neighbors']):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        radius = float(data['mean_radius'])
        area = float(data['mean_area'])
        perimeter = float(data['mean_perimeter'])
        texture = float(data['mean_texture'])
        smoothness = float(data['mean_smoothness'])
        n_neighbors = int(data['n_neighbors'])
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid input data type'}), 400

    x = df[["mean_radius", "mean_area", "mean_perimeter", "mean_texture", "mean_smoothness"]]
    y = df["diagnosis"]
    
    # It's better to train the model once and reuse it, but for simplicity here we train on each predict call.
    # For a production scenario, consider training the model when the app starts.
    modelo_classificador = KNeighborsClassifier(n_neighbors=n_neighbors)
    modelo_classificador.fit(x, y)
    
    prediction = modelo_classificador.predict([[radius, area, perimeter, texture, smoothness]])
    
    return jsonify({'prediction': int(prediction[0])}), 200


if __name__ == '__main__':
    app.run(debug=True)