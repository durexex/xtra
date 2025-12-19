import os
import json
import re
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report
import missingno as msno
import seaborn as sns

# Use a non-interactive backend for matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)

# Store dataframe in memory
df = None
MISSING_VALUE_TOKENS = ["Missing", "missing", "MISSING", "?"]

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
    try:
        # Work on a copy so we can coerce numeric-looking object columns and handle custom missing tokens
        describe_df = df.copy()
        describe_df = describe_df.replace(MISSING_VALUE_TOKENS, pd.NA)

        # Convert object columns to numeric when the majority of their non-null values are numeric
        for col in describe_df.select_dtypes(include=['object']).columns:
            numeric_col = pd.to_numeric(describe_df[col], errors='coerce')
            original_non_na = describe_df[col].notna().sum()
            converted_non_na = numeric_col.notna().sum()
            if original_non_na > 0 and converted_non_na >= 0.9 * original_non_na:
                describe_df[col] = numeric_col

        # Include all columns (numeric and categorical) in describe
        # Convert to JSON-safe primitives to avoid NaN/Timestamp serialization issues
        describe_result = describe_df.describe(include='all')
        safe_describe = json.loads(describe_result.to_json(date_format='iso'))
        return jsonify(safe_describe), 200
    except Exception as e:
        return jsonify({'error': f'Failed to compute describe: {str(e)}'}), 500

@app.route('/null-values', methods=['GET', 'POST'])
def get_null_values():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    
    custom_null_values = []
    columns_to_check = None
    if request.method == 'POST':
        data = request.get_json()
        if data:
            custom_null_values = data.get('custom_null_values', [])
            columns_to_check = data.get('columns_to_check')

    all_missing_tokens = MISSING_VALUE_TOKENS + custom_null_values
    
    # Create a temporary DataFrame for null value checking
    df_temp = df.copy() # Work on a copy to avoid side effects

    # Determine which columns to process
    columns_to_process = df.columns
    if columns_to_check and isinstance(columns_to_check, list):
        valid_columns = [col for col in columns_to_check if col in df.columns]
        if not valid_columns:
            return jsonify({'error': 'None of the selected columns exist in the dataframe'}), 400
        columns_to_process = valid_columns

    # First, replace all standard missing tokens and custom values with pd.NA
    # This ensures that values like '?' or 'missing' are handled uniformly.
    df_temp[columns_to_process] = df_temp[columns_to_process].replace(all_missing_tokens, pd.NA)

    # Second, iterate through custom null values and coerce them to the column's dtype
    # This is crucial for cases where a custom null is a number (e.g., 999) but stored as a string.
    for val in custom_null_values:
        for col in columns_to_process:
            try:
                # Attempt to convert the custom null value to the column's dtype
                coerced_val = pd.Series([val]).astype(df_temp[col].dtype).iloc[0]
                df_temp[col] = df_temp[col].replace(coerced_val, pd.NA)
            except (ValueError, TypeError):
                # If conversion fails (e.g., trying to cast 'N/A' to int), just continue
                continue
    
    # After all replacements, calculate the null counts for the processed columns.
    null_counts = df_temp[columns_to_process].isnull().sum()
    
    total = null_counts.to_dict()

    return jsonify(total), 200

@app.route('/fix-dataset', methods=['POST'])
def fix_dataset():
    """
    Normalize dataset by replacing known missing tokens and coercing object columns to Int64.
    """
    global df
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    try:
        fixed_df = df.copy()
        fixed_df = fixed_df.replace(MISSING_VALUE_TOKENS, pd.NA)

        # Force object columns into integer form (nullable) after coercing/rounding numerics
        for col in fixed_df.select_dtypes(include=['object']).columns:
            numeric_col = pd.to_numeric(fixed_df[col], errors='coerce')
            numeric_col = numeric_col.apply(lambda v: pd.NA if pd.isna(v) else int(round(v)))
            fixed_df[col] = pd.array(numeric_col, dtype='Int64')

        df = fixed_df
        metadata = _build_dataframe_info(df)

        return jsonify({
            'message': 'Dataset cleaned. Missing tokens replaced and object columns cast to Int64.',
            'columns': list(df.columns),
            'dtypes': metadata.get('dtypes'),
            'rows': metadata.get('rows')
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to fix dataset: {str(e)}'}), 500

@app.route('/download-reduced-dataset', methods=['GET'])
def download_reduced_dataset():
    """
    Return a CSV without selected columns (does not mutate the in-memory dataframe).
    """
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    filename = request.args.get('filename', 'dataset_reduced.csv')
    # Keep filename safe for header usage
    safe_filename = re.sub(r'[^A-Za-z0-9._-]', '_', filename) or 'dataset_reduced.csv'
    if not safe_filename.lower().endswith('.csv'):
        safe_filename += '.csv'

    drop_columns = [
        'Biopsy',
        'Cytology',  # keep spelling variant
        'Citology',  # dataset original spelling
        'Dx',
        'IUD (years)',
        'IUD: years',
        'STDs'
    ]

    try:
        reduced_df = df.drop(columns=drop_columns, errors='ignore')
        csv_buffer = io.StringIO()
        reduced_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return Response(
            csv_buffer.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={safe_filename}'}
        )
    except Exception as e:
        return jsonify({'error': f'Failed to generate reduced dataset: {str(e)}'}), 500

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
        # Use to_json/loads to coerce NaN/NaT into JSON-safe nulls
        safe_grouped = json.loads(grouped_describe.to_json(orient='index'))
        return jsonify(safe_grouped), 200
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

@app.route('/traintest', methods=['GET', 'POST'])
def get_traintest():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    n_neighbors = request.args.get('n_neighbors', default=3, type=int)
    y_col = request.args.get('y_col')
    predict = request.args.get('predict', 'false').lower() == 'true'

    if not y_col or y_col not in df.columns:
        return jsonify({'error': 'Invalid or missing y_col parameter'}), 400

    try:
        x = df.drop(columns=[y_col])
        y = df[y_col]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
        
        modelo_classificador = KNeighborsClassifier(n_neighbors=n_neighbors)
        modelo_classificador.fit(x_train, y_train)
        
        y_pred = modelo_classificador.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics_report = classification_report(y_test, y_pred, output_dict=True)
        
        html_response = f"""
        <div>
            <h3>Train-Test Split Details</h3>
            <p><strong>Training set size:</strong> {len(x_train)} samples</p>
            <p><strong>Test set size:</strong> {len(x_test)} samples</p>
            <p><strong>Accuracy:</strong> {accuracy:.4f}</p>
            <h4>Classification Metrics</h4>
            <pre>{json.dumps(metrics_report, indent=2)}</pre>
        </div>
        """

        if predict:
            if request.method != 'POST':
                return jsonify({'error': 'POST method required for prediction'}), 405
            
            data = request.get_json()
            if not data or 'predict_values' not in data:
                return jsonify({'error': 'Missing predict_values in request body'}), 400

            predict_values = data['predict_values']
            
            user_input = {}
            for col in x.columns:
                value = predict_values.get(col)
                if value is None or value == '':
                    return jsonify({'error': f'Missing value for {col}'}), 400
                try:
                    user_input[col] = [float(value)]
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid input for {col}: must be a number'}), 400
            
            user_df = pd.DataFrame.from_dict(user_input)
            prediction = modelo_classificador.predict(user_df)
            
            html_response += f"""
            <div>
                <h3>Prediction Result</h3>
                <p><strong>The prediction is:</strong> {prediction[0]}</p>
            </div>
            """
        
        return jsonify({'html': html_response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fix-nulls', methods=['POST'])
def fix_nulls():
    global df
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400

    custom_null_values = data.get('custom_null_values', [])
    columns_to_fix = data.get('columns_to_fix')

    if not columns_to_fix or not isinstance(columns_to_fix, list):
        return jsonify({'error': 'columns_to_fix must be a non-empty list'}), 400

    try:
        all_missing_tokens = MISSING_VALUE_TOKENS + custom_null_values

        # Create a copy to safely perform replacements and calculations
        df_temp = df.copy()

        # Step 1: Replace all known missing tokens with pd.NA in the specified columns
        df_temp[columns_to_fix] = df_temp[columns_to_fix].replace(all_missing_tokens, pd.NA)

        # Step 2: Coerce custom null values to the correct dtype for robust replacement
        for val in custom_null_values:
            for col in columns_to_fix:
                if col in df_temp.columns:
                    try:
                        coerced_val = pd.Series([val]).astype(df_temp[col].dtype).iloc[0]
                        df_temp[col] = df_temp[col].replace(coerced_val, pd.NA)
                    except (ValueError, TypeError):
                        continue
        
        # Step 3: Now that nulls are standardized, fill them with the median
        for col in columns_to_fix:
            if col in df.columns:
                # Coerce column to numeric, ignoring errors to handle non-numeric data
                numeric_col = pd.to_numeric(df_temp[col], errors='coerce')
                
                # Check if the column is of a numeric type before filling
                if pd.api.types.is_numeric_dtype(numeric_col):
                    median_val = numeric_col.median()
                    
                    # Fill NA values in the original DataFrame using the calculated median
                    # We use df_temp's null mask to fill the original df
                    df.loc[df_temp[col].isnull(), col] = median_val
                else:
                    # Skip non-numeric columns as median is not applicable
                    pass
        
        return jsonify({
            'message': 'Null values in selected columns fixed successfully using the median.',
            'download_endpoint': '/download-fixed-dataset'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-fixed-dataset', methods=['GET'])
def download_fixed_dataset():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    filename = request.args.get('filename', 'fixed_dataset.csv')
    # Keep filename safe for header usage
    safe_filename = re.sub(r'[^A-Za-z0-9._-]', '_', filename) or 'fixed_dataset.csv'
    if not safe_filename.lower().endswith('.csv'):
        safe_filename += '.csv'

    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return Response(
            csv_buffer.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={safe_filename}'}
        )
    except Exception as e:
        return jsonify({'error': f'Failed to generate fixed dataset: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
