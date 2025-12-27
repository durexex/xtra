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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils import ZeroToNaNTransformer, Log1pTransformer
import missingno as msno
import seaborn as sns
import math
import numpy as np
from pandas.plotting import scatter_matrix

# Use a non-interactive backend for matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Store dataframe in memory
df = None
MISSING_VALUE_TOKENS = ["Missing", "missing", "MISSING", "?"]


def calculate_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error, safely filtering out zero values in y_true.
    """
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    # Filter out cases where y_true is zero, as MAPE is undefined for them
    non_zero_mask = y_true_arr != 0
    if np.sum(non_zero_mask) == 0:
        return np.inf  # Or 0.0, depending on how you want to handle this edge case

    y_true_filt = y_true_arr[non_zero_mask]
    y_pred_filt = y_pred_arr[non_zero_mask]

    return float(np.mean(np.abs((y_true_filt - y_pred_filt) / y_true_filt)) * 100)


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
    app.logger.info(f"Request headers: {request.headers}")
    app.logger.info(f"Request files: {request.files}")
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
            return jsonify({'error': f'Error reading CSV file: {e}'}), 500
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


@app.route('/pearson-skewness', methods=['GET'])
def get_pearson_skewness():
    """
    Compute the second Pearson coefficient of skewness (A2) for numeric columns:
    A2 = 3 * (mean - median) / std.
    """
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    try:
        skew_df = df.copy()
        skew_df = skew_df.replace(MISSING_VALUE_TOKENS, pd.NA)

        # Convert mostly numeric object columns before computing stats
        for col in skew_df.select_dtypes(include=['object']).columns:
            numeric_col = pd.to_numeric(skew_df[col], errors='coerce')
            original_non_na = skew_df[col].notna().sum()
            converted_non_na = numeric_col.notna().sum()
            if original_non_na > 0 and converted_non_na >= 0.9 * original_non_na:
                skew_df[col] = numeric_col

        result = {col: None for col in skew_df.columns}
        numeric_df = skew_df.select_dtypes(include=['number'])

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                result[col] = None
                continue
            std = series.std()
            if std == 0 or pd.isna(std):
                result[col] = None
                continue
            mean = series.mean()
            median = series.median()
            skewness = 3 * (mean - median) / std
            try:
                result[col] = skewness.item() if hasattr(skewness, 'item') else float(skewness)
            except Exception:
                result[col] = float(skewness)

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Failed to compute Pearson skewness: {str(e)}'}), 500

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

    # Include common textual null markers beyond default missing tokens
    extra_null_tokens = ["NA", "Na", "na", "N/A", "n/a", "NULL", "Null", "null", "None", "none"]
    all_missing_tokens = list(dict.fromkeys(MISSING_VALUE_TOKENS + extra_null_tokens + custom_null_values))
    
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
    Normalize dataset by coercing selected object columns to Int64.
    """
    global df
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400
    data = request.get_json(silent=True) or {}
    requested_columns = data.get('columns')

    try:
        fixed_df = df.copy()

        if requested_columns and isinstance(requested_columns, list):
            target_cols = [col for col in requested_columns if col in fixed_df.columns]
            if not target_cols:
                return jsonify({'error': 'Nenhuma das colunas informadas existe no dataset.'}), 400
        else:
            target_cols = list(fixed_df.select_dtypes(include=['object']).columns)

        coerced_cols = []
        for col in target_cols:
            if col in fixed_df.columns and pd.api.types.is_object_dtype(fixed_df[col]):
                numeric_col = pd.to_numeric(fixed_df[col], errors='coerce')
                numeric_col = numeric_col.apply(lambda v: pd.NA if pd.isna(v) else int(round(v)))
                fixed_df[col] = pd.array(numeric_col, dtype='Int64')
                coerced_cols.append(col)

        df = fixed_df
        metadata = _build_dataframe_info(df)

        return jsonify({
            'message': 'Dataset cleaned. Object columns cast to Int64.',
            'columns': list(df.columns),
            'dtypes': metadata.get('dtypes'),
            'rows': metadata.get('rows'),
            'coerced_columns': coerced_cols
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

    # Accept custom columns via query param drop_columns (comma-separated) or repeatable params
    drop_param = request.args.get('drop_columns')
    drop_columns_list = request.args.getlist('drop_columns')

    if drop_param:
        drop_columns = [col for col in drop_param.split(',') if col]
    elif drop_columns_list:
        drop_columns = drop_columns_list
    
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

        # Calculate correlation using pandas corr and render it on the chart
        corr_matrix = df[[x_col, y_col]].corr()
        corr_value = corr_matrix.iloc[0, 1]
        corr_label = 'N/A' if pd.isna(corr_value) else f'{corr_value:.4f}'
        plt.gca().annotate(
            f'Correlation (pandas corr): {corr_label}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=9,
            ha='left',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Save plot to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode image to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        response_payload = {'image': image_base64}
        if not pd.isna(corr_value):
            response_payload['correlation'] = float(corr_value)
        else:
            response_payload['correlation'] = None
        
        return jsonify(response_payload), 200
        
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
    
    try:
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric columns available for histogram'}), 400

        # Choose bin count based on dataset size, capped for readability
        row_count = len(numeric_df)
        bins = max(10, min(50, int(math.sqrt(row_count)) if row_count > 0 else 10))

        # Determine a reasonable figure size based on the number of histograms
        num_cols = numeric_df.shape[1]
        cols_per_row = min(4, max(1, int(math.ceil(math.sqrt(num_cols)))))
        rows = int(math.ceil(num_cols / cols_per_row))
        figsize = (min(20, cols_per_row * 4), min(15, rows * 3.5))

        axes = numeric_df.hist(bins=bins, figsize=figsize)
        plt.tight_layout()

        # Retrieve the figure from the axes object
        fig = axes.ravel()[0].figure if hasattr(axes, 'ravel') else plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        return jsonify({
            'image': image_base64,
            'bins': int(bins),
            'figsize': [float(figsize[0]), float(figsize[1])],
            'columns': list(numeric_df.columns)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/correlation-matrix', methods=['GET'])
def get_correlation_matrix():
    """
    Generate a scatter matrix for all numeric columns to visualize pairwise correlations.
    """
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return jsonify({'error': 'No numeric columns available for correlation matrix'}), 400

    try:
        # Clean and guard against inf/nan-only frames
        safe_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()
        if safe_df.empty:
            return jsonify({'error': 'No numeric rows available for correlation matrix after cleaning'}), 400

        # Limit size if extremely wide to avoid huge plots
        max_cols = min(10, safe_df.shape[1])
        plot_df = safe_df.iloc[:, :max_cols]

        axes = scatter_matrix(plot_df, figsize=(min(16, 3 * max_cols), min(16, 3 * max_cols)), diagonal='hist')
        plt.tight_layout()
        fig = axes[0, 0].figure if hasattr(axes, '__getitem__') else plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return jsonify({
            'image': image_base64,
            'columns_used': list(plot_df.columns)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/categorize-column', methods=['POST'])
def categorize_column():
    """
    Solicita a coluna a ser categorizada, aplica ceil/1.5, cap superior via where,
    categoriza com pd.cut, faz split estratificado e envia histogramas/proporcoes.
    Ao final remove a coluna original, mantendo apenas a nova, e pode salvar em CSV.
    """
    global df
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    data = request.get_json(silent=True) or {}
    column = data.get('column')
    bins_param = data.get('bins')
    new_col = data.get('new_column') if column else None
    save_dir = data.get('save_dir') or data.get('save_path')
    save_name = data.get('save_name') or data.get('filename')
    allow_decimals_raw = data.get('allow_decimals')
    allow_decimals = False
    if isinstance(allow_decimals_raw, bool):
        allow_decimals = allow_decimals_raw
    elif allow_decimals_raw is not None:
        allow_decimals = str(allow_decimals_raw).strip().lower() in ('true', '1', 'yes', 'y', 'on')

    if not column:
        return jsonify({
            'error': 'Informe a coluna a ser categorizada em "column".',
            'available_columns': list(df.columns)
        }), 400
    if column not in df.columns:
        return jsonify({
            'error': f'Column {column} not found in dataframe.',
            'available_columns': list(df.columns)
        }), 400

    numeric_series = pd.to_numeric(df[column], errors='coerce')
    original_values = numeric_series.dropna()
    if original_values.empty:
        return jsonify({'error': 'Selected column has no numeric values to categorize'}), 400

    if bins_param is None:
        bins_estimate = int(math.sqrt(len(original_values))) if len(original_values) > 0 else 3
        bins = max(3, min(20, bins_estimate))
    else:
        try:
            bins = int(bins_param)
            if bins < 2:
                return jsonify({'error': 'O numero de bins deve ser pelo menos 2.'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Forneca um numero inteiro em "bins" para definir a categorizacao.'}), 400

    # Histograma da coluna original
    fig_original = plt.figure()
    original_values.plot.hist(bins=min(max(3, bins), 50), alpha=0.7)
    plt.title(f'{column} - Histograma original')
    plt.xlabel(column)
    plt.ylabel('Frequencia')
    buf_original = io.BytesIO()
    fig_original.savefig(buf_original, format='png')
    buf_original.seek(0)
    hist_original_b64 = base64.b64encode(buf_original.getvalue()).decode('utf-8')
    buf_original.close()
    plt.close(fig_original)

    # Ceil/1.5 e cap superior usando where
    rounded_series = np.ceil(numeric_series / 1.5)
    valid_rounded = rounded_series.dropna()
    if valid_rounded.empty:
        return jsonify({'error': 'Selected column has no numeric values to categorize'}), 400

    min_val = valid_rounded.min()
    max_val = valid_rounded.max()
    if min_val == max_val:
        return jsonify({'error': 'All numeric values are identical after ceil/1.5; cannot create bins.'}), 400

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    cap_value = bin_edges[-1]
    capped_series = rounded_series.where(rounded_series <= cap_value, cap_value)

    # Categorizar com pd.cut gerando codigos numericos (labels=False)
    try:
        cut_codes = pd.cut(
            capped_series.dropna(),
            bins=bin_edges,
            include_lowest=True,
            labels=False,
            duplicates='drop'
        )
    except ValueError as cut_err:
        return jsonify({'error': f'Nao foi possivel categorizar com os bins informados: {cut_err}'}), 400

    if cut_codes.dropna().empty:
        return jsonify({'error': 'Os bins escolhidos nao geraram categorias validas.'}), 400

    new_col_name = new_col or f"{column}_category"
    interval_index = pd.IntervalIndex.from_breaks(bin_edges, closed='right')

    # Serie de codigos inteiros base para mapear midpoints ou manter inteiro
    code_series = pd.Series(pd.NA, index=df.index, dtype="Int64")
    code_series.loc[cut_codes.index] = cut_codes.astype("Int64")

    def _interval_label(interval, use_decimals, is_first=False):
        """
        Build a human label for the bin.
        - Decimals: keep the native Interval string (open/closed per pd.cut).
        - Integers: produce non-overlapping integer ranges: [start, end],
          where the next bin starts at end+1.
        """
        if use_decimals:
            return str(interval)

        left = interval.left
        right = interval.right
        # First bin includes the lowest, others are left-open in pd.cut.
        start = math.floor(left) if is_first else math.floor(left) + 1
        end = math.floor(right)
        if start > end:
            start = end
        return f"[{int(start)}, {int(end)}]"

    if allow_decimals:
        midpoint_map = {int(idx): float((bin_edges[idx] + bin_edges[idx + 1]) / 2) for idx in range(len(bin_edges) - 1)}
        categorized = pd.Series(pd.NA, index=df.index, dtype="Float64")
        for code_val, midpoint in midpoint_map.items():
            categorized.loc[code_series == code_val] = midpoint
        category_mapping = {midpoint_map[int(idx)]: _interval_label(interval, True) for idx, interval in enumerate(interval_index)}
        categories = [_interval_label(interval, True) for interval in interval_index]
    else:
        categorized = code_series
        category_mapping = {int(idx): _interval_label(interval, False, is_first=(idx==0)) for idx, interval in enumerate(interval_index)}
        categories = [category_mapping[int(idx)] for idx in range(len(interval_index))]

    df[new_col_name] = categorized

    # Histograma da nova coluna categorizada
    fig_new = plt.figure(figsize=(8, 4.5))
    categorized.value_counts().sort_index().plot.bar()
    plt.title(f'{new_col_name} - Histograma categorizado')
    plt.xlabel('Bin (codigo numerico)')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    buf_new = io.BytesIO()
    fig_new.savefig(buf_new, format='png')
    buf_new.seek(0)
    hist_new_b64 = base64.b64encode(buf_new.getvalue()).decode('utf-8')
    buf_new.close()
    plt.close(fig_new)

    # Split estratificado conforme solicitado
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_index, test_index = next(splitter.split(df, df[new_col_name]))
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    except ValueError as split_err:
        return jsonify({'error': f'Nao foi possivel criar o split estratificado: {split_err}'}), 400

    def _proportions(target_df):
        counts = target_df[new_col_name].value_counts()
        denom = len(target_df) if len(target_df) > 0 else 1
        proportions_raw = counts / denom
        # Map valor (inteiro ou decimal) -> intervalo legivel (ou codigo inteiro)
        mapped = {}
        for idx, val in proportions_raw.items():
            key = idx
            if key not in category_mapping and isinstance(key, (int, float)) and key == int(key):
                key = int(key)
            mapped[str(category_mapping.get(key, key))] = float(val)
        return mapped

    proportions = {
        'train': _proportions(strat_train_set),
        'test': _proportions(strat_test_set),
        'full': _proportions(df),
        'sizes': {
            'train': int(len(strat_train_set)),
            'test': int(len(strat_test_set)),
            'full': int(len(df))
        }
    }

    # Remover coluna original e manter somente a nova
    df.drop(columns=[column], inplace=True)
    preview = df.head(50).to_dict(orient='records')

    # Solicitar nome/local de gravacao da nova base CSV
    saved_to = None
    save_prompt = None
    if save_dir and save_name:
        filename = save_name if str(save_name).lower().endswith('.csv') else f"{save_name}.csv"
        try:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, filename)
            df.to_csv(output_path, index=False)
            saved_to = output_path
        except Exception as save_err:
            save_prompt = f'Falha ao salvar CSV: {save_err}'
    else:
        save_prompt = 'Forneca "save_dir" (pasta) e "save_name" (nome do arquivo .csv) para salvar a nova base.'

    return jsonify({
        'message': f'Coluna {column} categorizada em {new_col_name} com ceil/1.5 e {bins} bins (valores acima do bin capados).',
        'new_column': new_col_name,
        'bins': int(bins),
        'hist_original': hist_original_b64,  # para comparacao no front
        'hist_new': hist_new_b64,
        'hist_before': hist_original_b64,    # compatibilidade com front atual
        'hist_after': hist_new_b64,
        'histograms': {  # bloco explicito para comparacao
            'original': hist_original_b64,
            'new': hist_new_b64
        },
        'proportions': proportions,
        'proportions_detail': {
            'column': new_col_name,
            'train': proportions['train'],
            'test': proportions['test'],
            'full': proportions['full']
        },
        'categories': categories,
        'category_mapping': category_mapping,
        'rounded_column': None,
        'split_method': 'stratified',
        'preview': preview,
        'columns': list(df.columns),
        'saved_to': saved_to,
        'save_prompt': save_prompt
    }), 200

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
        
        image_base64 = _save_plot_to_base64()
        
        return jsonify({'image': image_base64}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/linear-regression', methods=['POST'])
def run_linear_regression():
    """
    Execute multiple linear regression using all numeric columns except the chosen target.
    Drops the target from the training features, splits with iloc, fits the model and returns metrics.
    """
    global df
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    payload = request.get_json(silent=True) or {}
    target_column = payload.get('target')
    test_size = payload.get('test_size', 0.2)

    try:
        test_size = float(test_size)
    except (TypeError, ValueError):
        return jsonify({'error': 'test_size must be a number between 0 and 1 (e.g., 0.2).'}), 400

    if not 0 < test_size < 1:
        return jsonify({'error': 'test_size must be between 0 and 1 (e.g., 0.2).'}), 400

    if not target_column or target_column not in df.columns:
        return jsonify({'error': 'Invalid or missing target column'}), 400

    try:
        feature_df = df.drop(columns=[target_column])
        numeric_features = feature_df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        feature_columns = list(numeric_features.columns)

        if not feature_columns:
            return jsonify({'error': 'No numeric feature columns available after removing the target column.'}), 400

        target_series = pd.to_numeric(df[target_column], errors='coerce')

        working_df = numeric_features.copy()
        working_df['__target__'] = target_series
        working_df = working_df.dropna(subset=['__target__'])

        # Fill missing numeric feature values with column means to keep as many rows as possible.
        working_df[feature_columns] = working_df[feature_columns].fillna(working_df[feature_columns].mean())

        features = working_df[feature_columns].reset_index(drop=True)
        target = working_df['__target__'].reset_index(drop=True)

        if len(features) < 2:
            return jsonify({'error': 'Not enough data after cleaning to train and test the model.'}), 400

        indices = np.arange(len(features))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

        x_train = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]

        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        # Checa se a coluna de target é de tipo inteiro
        if pd.api.types.is_integer_dtype(df[target_column]):
            y_pred = np.round(y_pred).astype(int)

        metrics_payload = {
            'mean_squared_error': float(mean_squared_error(y_test, y_pred)),
            'mean_absolute_error': float(mean_absolute_error(y_test, y_pred)),
            'r2_score': float(r2_score(y_test, y_pred)),
            'mape': float(calculate_mape(y_test, y_pred))
        }

        preview_df = pd.DataFrame({
            'real': y_test.reset_index(drop=True),
            'predito': y_pred
        })
        preview_df['residuo'] = preview_df['real'] - preview_df['predito']
        sample_predictions = preview_df.head(30).to_dict(orient='records')

        return jsonify({
            'target': target_column,
            'feature_columns': feature_columns,
            'train_samples': int(len(train_idx)),
            'test_samples': int(len(test_idx)),
            'test_size': test_size,
            'metrics': metrics_payload,
            'sample_predictions': sample_predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/traintest', methods=['GET', 'POST'])
def get_traintest():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    n_neighbors = request.args.get('n_neighbors', default=3, type=int)
    y_col = request.args.get('y_col')
    predict = request.args.get('predict', 'false').lower() == 'true'
    scaling = request.args.get('scaling', 'none').lower()

    if not y_col or y_col not in df.columns:
        return jsonify({'error': 'Invalid or missing y_col parameter'}), 400

    try:
        x = df.drop(columns=[y_col])
        y = df[y_col]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
        
        scaler = None
        if scaling == 'standard':
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

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
            <p><strong>Scaling:</strong> {scaling}</p>
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
            
            if scaler:
                user_df = scaler.transform(user_df)

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
    strategy = str(data.get('strategy', 'median')).lower()

    if not columns_to_fix or not isinstance(columns_to_fix, list):
        return jsonify({'error': 'columns_to_fix must be a non-empty list'}), 400
    if strategy not in ['na', 'mean', 'median']:
        return jsonify({'error': 'strategy must be one of: na, mean, median'}), 400

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

        # Step 3: Apply fill strategy
        if strategy in ['mean', 'median']:
            for col in columns_to_fix:
                if col in df_temp.columns:
                    numeric_col = pd.to_numeric(df_temp[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(numeric_col):
                        fill_val = numeric_col.mean() if strategy == 'mean' else numeric_col.median()
                        df_temp.loc[df_temp[col].isnull(), col] = fill_val
                    else:
                        # Skip non-numeric columns for mean/median
                        continue
        # strategy == 'na' keeps pd.NA as is

        df = df_temp

        return jsonify({
            'message': f'Null values in selected columns fixed successfully using strategy: {strategy}.',
            'download_endpoint': '/download-fixed-dataset',
            'strategy': strategy
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


@app.route('/download-categorized-dataset', methods=['GET'])
def download_categorized_dataset():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    filename = request.args.get('filename', 'categorized_dataset.csv')
    drop_original = request.args.get('drop_original', 'false').lower() == 'true'
    source_column = request.args.get('source_column')
    target_column = request.args.get('target_column')

    safe_filename = re.sub(r'[^A-Za-z0-9._-]', '_', filename) or 'categorized_dataset.csv'
    if not safe_filename.lower().endswith('.csv'):
        safe_filename += '.csv'

    try:
        export_df = df.copy()
        if drop_original and source_column and target_column:
            # Drop the original column if target exists to preserve categorized view
            if source_column in export_df.columns and target_column in export_df.columns:
                export_df = export_df.drop(columns=[source_column])

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return Response(
            csv_buffer.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={safe_filename}'}
        )
    except Exception as e:
        return jsonify({'error': f'Failed to generate categorized dataset: {str(e)}'}), 500


def _save_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64
    

@app.route('/knn', methods=['POST'])
def knn():
    if df is None:
        return jsonify({'error': 'No dataframe loaded'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400

    target_column = data.get('target_column')
    zero_missing_cols = data.get('zero_missing_cols', [])
    log_cols = data.get('log_cols', [])
    imputer_strategy = data.get('imputer_strategy', 'median')
    scaler_option = data.get('scaler', 'RobustScaler')
    max_neighbors = data.get('max_neighbors', 10)

    if not target_column or target_column not in df.columns:
        return jsonify({'error': 'Invalid or missing target_column parameter'}), 400

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Create the pipeline
    steps = []
    if zero_missing_cols:
        steps.append(('zero_to_nan', ZeroToNaNTransformer(columns=zero_missing_cols)))
    
    steps.append(('imputer', SimpleImputer(strategy=imputer_strategy)))
    
    if log_cols:
        steps.append(('log1p', Log1pTransformer(columns=log_cols)))

    if scaler_option == 'StandardScaler':
        steps.append(('scaler', StandardScaler()))
    else:
        steps.append(('scaler', RobustScaler()))
    
    steps.append(('knn', KNeighborsClassifier()))

    pipeline = Pipeline(steps)

    # GridSearchCV
    param_grid = {
        'knn__n_neighbors': list(range(3, max_neighbors + 1)),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    try:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)

        return jsonify({
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
