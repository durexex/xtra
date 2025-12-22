import React, { useState } from 'react';
import './App.css';
import Modal from './components/Modal';
import RadioGroup from './components/RadioGroup';

function App() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [gridData, setGridData] = useState([]);
  const [gridTitle, setGridTitle] = useState('');
  const [htmlContent, setHtmlContent] = useState('');
  const [modalOpen, setModalOpen] = useState(false);
  const [groupByModalOpen, setGroupByModalOpen] = useState(false);
  const [scatterPlotModalOpen, setScatterPlotModalOpen] = useState(false);
  const [boxplotModalOpen, setBoxplotModalOpen] = useState(false);
  const [histogramModalOpen, setHistogramModalOpen] = useState(false);
  const [categorizeModalOpen, setCategorizeModalOpen] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterPlotImage, setScatterPlotImage] = useState('');
  const [scatterCorrelation, setScatterCorrelation] = useState(null);
  const [correlationMatrixImage, setCorrelationMatrixImage] = useState('');
  const [boxplotImage, setBoxplotImage] = useState('');
  const [boxplotColumn, setBoxplotColumn] = useState('');
  const [histogramImage, setHistogramImage] = useState('');
  const [histogramDetails, setHistogramDetails] = useState(null);
  const [categorizeColumn, setCategorizeColumn] = useState('');
  const [categorizeBins, setCategorizeBins] = useState('');
  const [categorizeAllowDecimals, setCategorizeAllowDecimals] = useState(false);
  const [categorizeHistBefore, setCategorizeHistBefore] = useState('');
  const [categorizeHistAfter, setCategorizeHistAfter] = useState('');
  const [categorizeInfo, setCategorizeInfo] = useState(null);
  const [categorizeProportions, setCategorizeProportions] = useState(null);
  const [categorizeSplitMethod, setCategorizeSplitMethod] = useState('');
  const [knnModalOpen, setKnnModalOpen] = useState(false);
  const [maxNeighbors, setMaxNeighbors] = useState(10);
  const [knnY, setKnnY] = useState('');
  const [knnScaling, setKnnScaling] = useState('RobustScaler');
  const [imputerStrategy, setImputerStrategy] = useState('median');
  const [zeroMissingCols, setZeroMissingCols] = useState({});
  const [logCols, setLogCols] = useState({});
  const [linearModalOpen, setLinearModalOpen] = useState(false);
  const [linearTargetColumn, setLinearTargetColumn] = useState('');
  const [linearTestSize, setLinearTestSize] = useState(0.2);
  const [linearMetrics, setLinearMetrics] = useState(null);
  const [linearPredictions, setLinearPredictions] = useState([]);
  const [nullValuesModalOpen, setNullValuesModalOpen] = useState(false);
  const [customNullValues, setCustomNullValues] = useState([]);
  const [currentCustomNullValue, setCurrentCustomNullValue] = useState('');
  const [selectedColumnsForNullCheck, setSelectedColumnsForNullCheck] = useState({});
  const [fixStrategy, setFixStrategy] = useState('median');
  const [predictValues, setPredictValues] = useState({});

  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [infoData, setInfoData] = useState(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [infoError, setInfoError] = useState(null);
  const [fixModalOpen, setFixModalOpen] = useState(false);
  const [fixingDataset, setFixingDataset] = useState(false);
  const [fixMessage, setFixMessage] = useState('');
  const [convertModalOpen, setConvertModalOpen] = useState(false);
  const [selectedColumnsForConvert, setSelectedColumnsForConvert] = useState({});
  const [reducedModalOpen, setReducedModalOpen] = useState(false);
  const [selectedColumnsForReduced, setSelectedColumnsForReduced] = useState({});

  const resetLinearRegressionOutput = () => {
    setLinearMetrics(null);
    setLinearPredictions([]);
  };

  const formatMetric = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '-';
    }
    return Number(value).toFixed(4);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file first.");
      return;
    }
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setColumns(data.columns);
        setModalOpen(false);
        setGridData([]);
        setGridTitle('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setCorrelationMatrixImage('');
        setBoxplotImage('');
        setBoxplotColumn('');
        setHistogramImage('');
        setHistogramDetails(null);
        setHtmlContent('');
        setInfoData(null);
        setInfoError(null);
        setCategorizeColumn('');
        setCategorizeBins('');
        setCategorizeAllowDecimals(false);
        setCategorizeModalOpen(false);
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        resetLinearRegressionOutput();
        setLinearTargetColumn('');
        setLinearTestSize(0.2);
        setLinearModalOpen(false);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred while uploading the file.');
    }
  };

  const openModal = () => setModalOpen(true);
  const closeModal = () => setModalOpen(false);
  const openGroupByModal = () => setGroupByModalOpen(true);
  const closeGroupByModal = () => setGroupByModalOpen(false);
  const openScatterPlotModal = () => setScatterPlotModalOpen(true);
  const closeScatterPlotModal = () => setScatterPlotModalOpen(false);
  const openBoxplotModal = () => setBoxplotModalOpen(true);
  const closeBoxplotModal = () => setBoxplotModalOpen(false);
  const openHistogramModal = () => setHistogramModalOpen(true);
  const closeHistogramModal = () => setHistogramModalOpen(false);
  const openCategorizeModal = () => setCategorizeModalOpen(true);
  const closeCategorizeModal = () => setCategorizeModalOpen(false);
  const openKnnModal = () => {
    setKnnModalOpen(true);
  };
  const closeKnnModal = () => setKnnModalOpen(false);
  const openLinearModal = () => setLinearModalOpen(true);
  const closeLinearModal = () => setLinearModalOpen(false);

  const closeInfoModal = () => {
    setInfoModalOpen(false);
    setInfoLoading(false);
  };
  const openConvertModal = () => {
    const allColumns = columns.reduce((acc, col) => {
      acc[col] = true;
      return acc;
    }, {});
    setSelectedColumnsForConvert(allColumns);
    setConvertModalOpen(true);
  };
  const closeConvertModal = () => setConvertModalOpen(false);
  const openReducedModal = () => {
    const noneSelected = {};
    setSelectedColumnsForReduced(noneSelected);
    setReducedModalOpen(true);
  };
  const closeReducedModal = () => setReducedModalOpen(false);

  const fetchHead = async () => {
    try {
      const response = await fetch('http://localhost:5000/head');
      const data = await response.json();
      if (response.ok) {
        setGridData(data);
        setGridTitle('DataFrame Head:');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setBoxplotImage('');
        setBoxplotColumn('');
        setCorrelationMatrixImage('');
        setHistogramImage('');
        setHistogramDetails(null);
        setHtmlContent('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        resetLinearRegressionOutput();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching head data:', error);
      alert('An error occurred while fetching the head data.');
    }
  };

  const fetchDescribe = async () => {
    try {
      const response = await fetch('http://localhost:5000/describe');
      const data = await response.json();
      if (response.ok) {
        // Transform the data to be easily rendered in a table
        const transformedData = Object.keys(data[Object.keys(data)[0]]).map(index => {
          const row = { index };
          Object.keys(data).forEach(col => {
            row[col] = data[col][index];
          });
          return row;
        });
        setGridData(transformedData);
        setGridTitle('DataFrame Describe:');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setBoxplotImage('');
        setBoxplotColumn('');
        setCorrelationMatrixImage('');
        setHistogramImage('');
        setHistogramDetails(null);
        setHtmlContent('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        resetLinearRegressionOutput();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching describe data:', error);
      alert('An error occurred while fetching the describe data.');
    }
  };

  const openNullValuesModal = () => setNullValuesModalOpen(true);
  const closeNullValuesModal = () => setNullValuesModalOpen(false);

  const addCustomNullValue = () => {
    if (currentCustomNullValue && !customNullValues.includes(currentCustomNullValue)) {
      setCustomNullValues([...customNullValues, currentCustomNullValue]);
      setCurrentCustomNullValue('');
    }
  };

  const clearCustomNullValues = () => {
    setCustomNullValues([]);
  };

  const handleConvertCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setSelectedColumnsForConvert(prev => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleConvertSelectAll = (checked) => {
    const allColumns = columns.reduce((acc, col) => {
      acc[col] = checked;
      return acc;
    }, {});
    setSelectedColumnsForConvert(allColumns);
  };

  const handleReducedCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setSelectedColumnsForReduced(prev => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleReducedSelectAll = (checked) => {
    const allColumns = columns.reduce((acc, col) => {
      acc[col] = checked;
      return acc;
    }, {});
    setSelectedColumnsForReduced(allColumns);
  };

  const handleCheckboxChange = (event) => {
    const { name, checked } = event.target;
    setSelectedColumnsForNullCheck(prevState => ({
      ...prevState,
      [name]: checked,
    }));
  };


  const fetchNullValues = async () => {
    const selectedColumns = Object.keys(selectedColumnsForNullCheck).filter(col => selectedColumnsForNullCheck[col]);
    
    if (selectedColumns.length === 0) {
      alert("Please select at least one column to check for null values.");
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/null-values', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ custom_null_values: customNullValues, columns_to_check: selectedColumns }),
      });
      const data = await response.json();
      if (response.ok) {
        const transformedData = Object.keys(data).map(key => ({
          Column: key,
          Null_Values: data[key]
        }));
        setGridData(transformedData);
        setGridTitle('Valores Nulos por Coluna:');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setHistogramImage('');
        setHistogramDetails(null);
        setCorrelationMatrixImage('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        setHtmlContent('');
        resetLinearRegressionOutput();
        closeNullValuesModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching null values:', error);
      alert('An error occurred while fetching null values.');
    }
  };

  const handleFixNullValues = async () => {
    const selectedColumns = Object.keys(selectedColumnsForNullCheck).filter(col => selectedColumnsForNullCheck[col]);
    
    if (selectedColumns.length === 0) {
      alert("Please select at least one column to fix null values.");
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/fix-nulls', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ custom_null_values: customNullValues, columns_to_fix: selectedColumns, strategy: fixStrategy }),
      });
      const data = await response.json();
      if (response.ok) {
        alert(data.message);
        closeNullValuesModal();
        
        // After fixing, prompt for download
        const defaultName = 'fixed_dataset.csv';
        const fileName = window.prompt('Enter the filename for the fixed dataset:', defaultName);
        if (fileName) { // Proceed if user doesn't cancel the prompt
            const trimmed = fileName.trim() || defaultName;
            const finalName = trimmed.toLowerCase().endsWith('.csv') ? trimmed : `${trimmed}.csv`;

            // Use the download endpoint from the response
            const downloadResponse = await fetch(`http://localhost:5000${data.download_endpoint}?filename=${encodeURIComponent(finalName)}`);
            
            if (downloadResponse.ok) {
                const blob = await downloadResponse.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = finalName;
                document.body.appendChild(link);
                link.click();
                link.remove();
                window.URL.revokeObjectURL(url);
            } else {
                const errorData = await downloadResponse.json().catch(() => ({}));
                alert(`Error downloading file: ${errorData.error || 'Unknown error'}`);
            }
        }
        fetchHead(); // Refresh the data grid
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fixing null values:', error);
      alert('An error occurred while fixing null values.');
    }
  };


  const fetchDataframeInfo = async () => {
    setInfoModalOpen(true);
    setInfoLoading(true);
    setInfoError(null);
    try {
      const response = await fetch('http://localhost:5000/dataframe-info');
      const data = await response.json();
      if (response.ok) {
        setInfoData(data);
      } else {
        setInfoError(data.error || 'Failed to fetch dataframe info.');
      }
    } catch (error) {
      console.error('Error fetching dataframe info:', error);
      setInfoError('An error occurred while fetching dataframe info.');
    } finally {
      setInfoLoading(false);
    }
  };

  const handleFixDataset = async () => {
    if (columns.length === 0) {
      alert("Please upload a dataset first.");
      return;
    }

    const selectedColumns = Object.keys(selectedColumnsForConvert).filter(col => selectedColumnsForConvert[col]);
    if (selectedColumns.length === 0) {
      alert("Selecione ao menos uma coluna para converter.");
      return;
    }

    setFixingDataset(true);
    try {
      const response = await fetch('http://localhost:5000/fix-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ columns: selectedColumns })
      });
      const data = await response.json();
      if (response.ok) {
        setColumns(data.columns || columns);
        setFixMessage(data.message || 'Dataset fixed successfully.');
        setFixModalOpen(true);
        closeConvertModal();
        setGridData([]);
        setGridTitle('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setBoxplotImage('');
        setBoxplotColumn('');
        setHistogramImage('');
        setHistogramDetails(null);
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCorrelationMatrixImage('');
        setHtmlContent('');
        if (data.info) {
          setInfoData(data.info);
        }
      } else {
        alert(`Error: ${data.error || 'Failed to fix dataset.'}`);
      }
    } catch (error) {
      console.error('Error fixing dataset:', error);
      alert('An error occurred while fixing the dataset.');
    } finally {
      setFixingDataset(false);
    }
  };

  const handleDownloadFixedDataset = async () => {
    try {
      const response = await fetch('http://localhost:5000/download-fixed-dataset');
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        alert(`Error: ${errorData.error || 'Failed to download fixed dataset.'}`);
        return;
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'fixed_dataset.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      setFixModalOpen(false);
    } catch (error) {
      console.error('Error downloading fixed dataset:', error);
      alert('An error occurred while downloading the fixed dataset.');
    }
  };

  const closeFixModal = () => setFixModalOpen(false);

  const handleGroupBySubmit = async (e) => {
    e.preventDefault();
    if (!selectedColumn) {
      alert("Please select a column.");
      return;
    }
    try {
      const response = await fetch(`http://localhost:5000/groupby?column=${selectedColumn}`);
      const data = await response.json();
      if (response.ok) {
        // The backend now returns a flat dictionary where keys are the groups.
        // We can convert this into an array of objects for the grid.
        const transformedData = Object.keys(data).map(groupName => {
          return { Group: groupName, ...data[groupName] };
        });
        setGridData(transformedData);
        setGridTitle(`Grouped Describe by ${selectedColumn}:`);
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setHistogramImage('');
        setHistogramDetails(null);
        setCorrelationMatrixImage('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        setHtmlContent('');
        resetLinearRegressionOutput();
        closeGroupByModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching group by data:', error);
      alert('An error occurred while fetching the group by data.');
    }
  };

  const handleBoxplotSubmit = async (column) => {
    if (!column) {
      alert("Please select a column.");
      return;
    }
    try {
      const response = await fetch(`http://localhost:5000/boxplot?column=${column}`);
      const data = await response.json();
      if (response.ok) {
        setBoxplotImage(`data:image/png;base64,${data.image}`);
        setBoxplotColumn(column);
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setHistogramImage('');
        setHistogramDetails(null);
        setCorrelationMatrixImage('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        resetLinearRegressionOutput();
        closeBoxplotModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching boxplot data:', error);
      alert('An error occurred while fetching the boxplot data.');
    }
  };

  const downloadImageWithName = (dataUrl, suggestedName) => {
    if (!dataUrl) {
      return;
    }
    const defaultName = suggestedName || 'download.png';
    const input = window.prompt('Digite o nome do arquivo para salvar:', defaultName);
    if (input === null) {
      return;
    }
    const trimmed = (input || '').trim() || defaultName;
    const finalName = trimmed.toLowerCase().endsWith('.png') ? trimmed : `${trimmed}.png`;

    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = finalName;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleBoxplotDownload = () => {
    const suggestedName = `boxplot_${boxplotColumn || 'plot'}.png`;
    downloadImageWithName(boxplotImage, suggestedName);
  };

  const handleHistogramSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/histogram');
      const data = await response.json();
      if (response.ok) {
        setHistogramImage(`data:image/png;base64,${data.image}`);
        setHistogramDetails({
          bins: data.bins,
          figsize: data.figsize,
          columns: data.columns,
        });
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setBoxplotImage('');
        setBoxplotColumn('');
        setCorrelationMatrixImage('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        resetLinearRegressionOutput();
        closeHistogramModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching histogram data:', error);
      alert('An error occurred while fetching the histogram data.');
    }
  };

  const handleCorrelationMatrixSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/correlation-matrix');
      const data = await response.json();
      if (response.ok) {
        setCorrelationMatrixImage(`data:image/png;base64,${data.image}`);
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setBoxplotImage('');
        setBoxplotColumn('');
        setHistogramImage('');
        setHistogramDetails(null);
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        resetLinearRegressionOutput();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching correlation matrix:', error);
      alert('An error occurred while fetching the correlation matrix.');
    }
  };

  const handleScatterPlotSubmit = async (xCol, yCol) => {
    if (!xCol || !yCol) {
      alert("Please select both x and y columns.");
      return;
    }
    if (!columns.includes(xCol) || !columns.includes(yCol)) {
      alert("The selected columns are not available in the dataset.");
      return;
    }
    try {
      const response = await fetch(`http://localhost:5000/scatterplot?x_col=${encodeURIComponent(xCol)}&y_col=${encodeURIComponent(yCol)}`);
      const data = await response.json();
      if (response.ok) {
        setScatterPlotImage(`data:image/png;base64,${data.image}`);
        setScatterCorrelation(data.correlation ?? null);
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setBoxplotImage('');
        setBoxplotColumn('');
        setHistogramImage('');
        setHistogramDetails(null);
        setCorrelationMatrixImage('');
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        resetLinearRegressionOutput();
        closeScatterPlotModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching scatter plot data:', error);
      alert('An error occurred while fetching the scatter plot data.');
    }
  };

  const handleScatterPlot3dSubmit = async () => {
    try {
      const response = await fetch(`http://localhost:5000/scatterplot3d?x_col=mean_area&y_col=mean_perimeter&z_col=mean_radius`);
      const data = await response.json();
      if (response.ok) {
        setScatterPlotImage(`data:image/png;base64,${data.image}`);
        setScatterCorrelation(null);
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setHistogramImage('');
        setHistogramDetails(null);
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        setCorrelationMatrixImage('');
        resetLinearRegressionOutput();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching 3D scatter plot data:', error);
      alert('An error occurred while fetching the 3D scatter plot data.');
    }
  };

  const handleLinearRegressionSubmit = async (e) => {
    e.preventDefault();
    if (!linearTargetColumn) {
      alert("Selecione a coluna target.");
      return;
    }

    const parsedTestSize = parseFloat(linearTestSize);
    if (Number.isNaN(parsedTestSize) || parsedTestSize <= 0 || parsedTestSize >= 1) {
      alert("Defina um test_size entre 0 e 1 (ex: 0.2).");
      return;
    }

    const payload = {
      target: linearTargetColumn,
      test_size: parsedTestSize,
    };

    try {
      resetLinearRegressionOutput();
      const response = await fetch('http://localhost:5000/linear-regression', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (response.ok) {
        const metricsPayload = {
          target: data.target,
          featureColumns: data.feature_columns || [],
          trainSamples: data.train_samples,
          testSamples: data.test_samples,
          testSize: data.test_size,
          ...data.metrics,
        };
        setLinearMetrics(metricsPayload);
        setLinearPredictions(data.sample_predictions || []);
        setGridData([]);
        setGridTitle('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setHistogramImage('');
        setHistogramDetails(null);
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        setCorrelationMatrixImage('');
        setBoxplotImage('');
        setBoxplotColumn('');
        setHtmlContent('');
        closeLinearModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error running linear regression:', error);
      alert('An error occurred while running linear regression.');
    }
  };


  const fetchTrainTest = async (n_neighbors, y_col, scaling, imputerStrategy, zeroMissing, log) => {
    try {
      const response = await fetch('http://localhost:5000/knn', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          target_column: y_col,
          zero_missing_cols: zeroMissing,
          log_cols: log,
          imputer_strategy: imputerStrategy,
          scaler: scaling,
          max_neighbors: n_neighbors
        })
      });
      const data = await response.json();
      
      if (response.ok) {
        setHtmlContent(JSON.stringify(data, null, 2));
        setGridData([]);
        setGridTitle('');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setHistogramImage('');
        setHistogramDetails(null);
        setCategorizeHistBefore('');
        setCategorizeHistAfter('');
        setCategorizeInfo(null);
        setCategorizeProportions(null);
        setCategorizeSplitMethod('');
        setCorrelationMatrixImage('');
        resetLinearRegressionOutput();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching KNN (test & train) data:', error);
      alert('An error occurred while fetching the KNN (test & train) data.');
    }
  };

  const handleKnnSubmit = (e) => {
    e.preventDefault();
    if (!knnY) {
      alert("Please select a Y column.");
      return;
    }

    const zeroMissing = Object.keys(zeroMissingCols).filter(col => zeroMissingCols[col]);
    const log = Object.keys(logCols).filter(col => logCols[col]);

    fetchTrainTest(maxNeighbors, knnY, knnScaling, imputerStrategy, zeroMissing, log);
    closeKnnModal();
  };



  const renderProportionTable = (label, data, categories) => {
    const keys = (categories && categories.length ? categories : Object.keys(data || {})) || [];
    return (
      <div style={{ minWidth: 180 }}>
        <h4>{label}</h4>
        <table className="grid">
          <thead>
            <tr>
              <th>Categoria</th>
              <th>Proporcao</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{data && data[k] !== undefined ? data[k].toFixed(3) : '0.000'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const handleCategorizeSubmit = async (e) => {
    e.preventDefault();
    if (!categorizeColumn) {
      alert("Selecione uma coluna para categorizar.");
      return;
    }

    const payload = { column: categorizeColumn, allow_decimals: categorizeAllowDecimals };
    if (categorizeBins) {
      payload.bins = Number(categorizeBins);
    }

    try {
      const response = await fetch('http://localhost:5000/categorize-column', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (response.ok) {
        setColumns(data.columns || columns);
        setGridData(data.preview || []);
        setGridTitle(data.message || 'Coluna categorizada.');
        setScatterPlotImage('');
        setScatterCorrelation(null);
        setCorrelationMatrixImage('');
        setBoxplotImage('');
        setBoxplotColumn('');
        setHistogramImage('');
        setHistogramDetails(null);
        setHtmlContent('');
        setCategorizeColumn('');
        setCategorizeBins('');
        setCategorizeAllowDecimals(false);
        setCategorizeHistBefore(data.hist_before ? `data:image/png;base64,${data.hist_before}` : '');
        setCategorizeHistAfter(data.hist_after ? `data:image/png;base64,${data.hist_after}` : '');
        setCategorizeInfo({
          bins: data.bins,
          newColumn: data.new_column,
          categories: data.categories,
          roundedColumn: data.rounded_column,
        });
        setCategorizeProportions(data.proportions || null);
        setCategorizeSplitMethod(data.split_method || '');
        resetLinearRegressionOutput();
        closeCategorizeModal();
        alert(data.message);

        // Prompt to save the categorized dataset
        const defaultName = 'categorized_dataset.csv';
        const fileName = window.prompt('Digite o nome do arquivo para salvar o dataset categorizado (coluna original sera removida):', defaultName);
        if (fileName !== null) {
          const trimmed = (fileName || '').trim() || defaultName;
          const finalName = trimmed.toLowerCase().endsWith('.csv') ? trimmed : `${trimmed}.csv`;
          try {
            const params = new URLSearchParams({
              filename: finalName,
              drop_original: 'true',
              source_column: categorizeColumn,
              target_column: categorizeInfo?.newColumn || ''
            });
            const downloadResponse = await fetch(`http://localhost:5000/download-categorized-dataset?${params.toString()}`);
            if (downloadResponse.ok) {
              const blob = await downloadResponse.blob();
              const url = window.URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = finalName;
              document.body.appendChild(link);
              link.click();
              link.remove();
              window.URL.revokeObjectURL(url);
            } else {
              const errorData = await downloadResponse.json().catch(() => ({}));
              alert(`Error downloading categorized dataset: ${errorData.error || 'Unknown error'}`);
            }
          } catch (downloadErr) {
            console.error('Error downloading categorized dataset:', downloadErr);
            alert('Ocorreu um erro ao baixar o dataset categorizado.');
          }
        }
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error categorizing column:', error);
      alert('An error occurred while categorizing the column.');
    }
  };



  const handleDownloadReducedDataset = async () => {
    if (columns.length === 0) {
      alert("Please upload a dataset first.");
      return;
    }
    const columnsToDrop = Object.keys(selectedColumnsForReduced).filter(col => selectedColumnsForReduced[col]);
    if (columnsToDrop.length === 0) {
      alert('Selecione ao menos uma coluna para excluir.');
      return;
    }

    const defaultName = 'dataset_reduced.csv';
    const fileName = window.prompt('Digite o nome do arquivo para salvar (sugerido: dataset_reduced.csv):', defaultName);
    if (fileName === null) {
      return;
    }
    const trimmed = fileName.trim() || defaultName;
    const finalName = trimmed.toLowerCase().endsWith('.csv') ? trimmed : `${trimmed}.csv`;

    try {
      const params = new URLSearchParams();
      params.append('filename', finalName);
      params.append('drop_columns', columnsToDrop.join(','));
      const response = await fetch(`http://localhost:5000/download-reduced-dataset?${params.toString()}`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        alert(`Error: ${errorData.error || 'Failed to download reduced dataset.'}`);
        return;
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = finalName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      closeReducedModal();
    } catch (error) {
      console.error('Error downloading reduced dataset:', error);
      alert('An error occurred while downloading the reduced dataset.');
    }
  };

  return (
    <div className="App">
      <div className="sidebar">
        <h2>Menu</h2>
        <button onClick={openModal}>Load File</button>
        <button onClick={fetchHead} disabled={columns.length === 0}>Head</button>
        <button onClick={fetchDescribe} disabled={columns.length === 0}>Describe</button>
        <button onClick={openGroupByModal} disabled={columns.length === 0}>Group By</button>
        <button onClick={openNullValuesModal} disabled={columns.length === 0}>Valores Nulos</button>
        <button onClick={fetchDataframeInfo} disabled={columns.length === 0}>Dataframe Info</button>
        <button onClick={openConvertModal} disabled={columns.length === 0 || fixingDataset}>
          {fixingDataset ? 'Convertendo...' : 'Converter str em num'}
        </button>
        <button onClick={openReducedModal} disabled={columns.length === 0}>Salvar reduzido</button>
        <button onClick={openBoxplotModal} disabled={columns.length === 0}>Boxplot</button>
        <button onClick={openHistogramModal} disabled={columns.length === 0}>Histogram</button>
        <button onClick={handleCorrelationMatrixSubmit} disabled={columns.length === 0}>Matriz Correlacao</button>
        <button onClick={openCategorizeModal} disabled={columns.length === 0}>Categorizar coluna</button>
        <button onClick={openScatterPlotModal} disabled={columns.length === 0}>Scatter Plot</button>
        <button onClick={handleScatterPlot3dSubmit} disabled={true}>3D Scatter Plot</button>
        <button onClick={openKnnModal} disabled={columns.length === 0}>KNN</button>
        <button onClick={openLinearModal} disabled={columns.length === 0}>Regressao Linear Multipla</button>
      </div>
      <div className="content">
        <div className="page-shell">
          <div className="page-card">
            <h1>Dashboard</h1>
            {gridData.length > 0 && (
              <div className="table-container">
                <h2>{gridTitle}</h2>
                <table className="grid">
                  <thead>
                    <tr>
                      {Object.keys(gridData[0]).map((key) => (
                        <th key={key}>{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {gridData.map((row, index) => (
                      <tr key={index}>
                        {Object.values(row).map((val, i) => (
                          <td key={i}>{val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {scatterPlotImage && (
              <div>
                <h2>Scatter Plot:</h2>
                <p>Correlacao (pandas corr): {scatterCorrelation === null ? 'N/A' : scatterCorrelation.toFixed(4)}</p>
                <img src={scatterPlotImage} alt="Scatter Plot" />
              </div>
            )}
           {correlationMatrixImage && (
              <div>
                <h2>Matriz de Correlacao (scatter_matrix):</h2>
                <img src={correlationMatrixImage} alt="Matriz de correlacao" />
              </div>
            )}
            {boxplotImage && (
              <div>
                <h2>Boxplot:</h2>
               <p className="muted">
                 Clique no boxplot para salvar (sugestão: {`boxplot_${boxplotColumn || 'plot'}.png`}).
               </p>
               <img
                 src={boxplotImage}
                 alt="Boxplot"
                 style={{ cursor: 'pointer' }}
                 onClick={handleBoxplotDownload}
               />
             </div>
           )}
           {histogramImage && (
              <div>
                <h2>Histogramas:</h2>
                {histogramDetails && (
                  <p>
                    Bins: {histogramDetails.bins ?? '-'} | Figsize: (
                    {histogramDetails.figsize ? histogramDetails.figsize[0] : '-'} x {histogramDetails.figsize ? histogramDetails.figsize[1] : '-'}
                    ) | Colunas: {histogramDetails.columns ? histogramDetails.columns.length : '-'}
                  </p>
                )}
                <img src={histogramImage} alt="Histogram" />
              </div>
            )}
            {categorizeHistBefore && categorizeHistAfter && (
              <div style={{ marginTop: '1rem' }}>
                <h2>Histogramas (antes e apos categorizar):</h2>
                {categorizeInfo && (
                  <p>
                    Bins usados: {categorizeInfo.bins ?? '-'} | Nova coluna (categorias): {categorizeInfo.newColumn || '-'} | Coluna arredondada: {categorizeInfo.roundedColumn || '-'} | Categorias: {categorizeInfo.categories ? categorizeInfo.categories.length : 0}
                  </p>
                )}
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  <div>
                    <h4>Antes</h4>
                    <img src={categorizeHistBefore} alt="Histograma antes da categorizacao" style={{ maxWidth: '320px' }} />
                  </div>
                  <div>
                    <h4>Depois</h4>
                    <img src={categorizeHistAfter} alt="Histograma apos categorizacao" style={{ maxWidth: '320px' }} />
                  </div>
                </div>
                {categorizeProportions && (
                  <div style={{ marginTop: '12px' }}>
                    <h3>Proporcoes estratificadas (StratifiedShuffleSplit)</h3>
                    <p>
                      Tamanhos - Train: {categorizeProportions.sizes?.train ?? '-'} | Test: {categorizeProportions.sizes?.test ?? '-'} | Full: {categorizeProportions.sizes?.full ?? '-'}
                      {categorizeSplitMethod && ` | Metodo: ${categorizeSplitMethod}`}
                    </p>
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                      {renderProportionTable('Train', categorizeProportions.train, categorizeInfo?.categories)}
                      {renderProportionTable('Test', categorizeProportions.test, categorizeInfo?.categories)}
                      {renderProportionTable('Full', categorizeProportions.full, categorizeInfo?.categories)}
                    </div>
                  </div>
                )}
              </div>
            )}
            {gridData.length > 0 && gridTitle.includes('bucketized') && (
              <div style={{ marginTop: '1rem' }}>
                <p><strong>{gridTitle}</strong></p>
              </div>
            )}
            {linearMetrics && (
              <div className="table-container">
                <h2>Regressao Linear Multipla</h2>
                <p>Target: {linearMetrics.target || '-'}</p>
                <p>
                  Colunas de treino:{' '}
                  {linearMetrics.featureColumns && linearMetrics.featureColumns.length
                    ? linearMetrics.featureColumns.join(', ')
                    : 'Nenhuma coluna numerica disponivel'}
                </p>
                <p>
                  Train samples: {linearMetrics.trainSamples ?? '-'} | Test samples: {linearMetrics.testSamples ?? '-'} | Test size:{' '}
                  {formatMetric(linearMetrics.testSize)}
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px' }}>
                  <div><strong>mean_squared_error:</strong> {formatMetric(linearMetrics.mean_squared_error)}</div>
                  <div><strong>mean_absolute_error:</strong> {formatMetric(linearMetrics.mean_absolute_error)}</div>
                  <div><strong>r2_score:</strong> {formatMetric(linearMetrics.r2_score)}</div>
                  <div><strong>MAPE (%):</strong> {formatMetric(linearMetrics.mape)}</div>
                </div>
              </div>
            )}
            {linearPredictions.length > 0 && (
              <div className="table-container">
                <h3>Previsto x Real (amostra)</h3>
                <table className="grid">
                  <thead>
                    <tr>
                      <th>Real</th>
                      <th>Previsto</th>
                      <th>Residuo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {linearPredictions.map((row, idx) => (
                      <tr key={idx}>
                        <td>{formatMetric(row.real)}</td>
                        <td>{formatMetric(row.predito)}</td>
                        <td>{formatMetric(row.residuo)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {htmlContent && (
               <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
            )}
          </div>
        </div>
      </div>

      {modalOpen && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closeModal}>&times;</span>
            <h2>Upload CSV File</h2>
            <form onSubmit={handleSubmit}>
              <input type="file" onChange={handleFileChange} accept=".csv" />
              <button type="submit">Upload</button>
            </form>
          </div>
        </div>
      )}

      {groupByModalOpen && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closeGroupByModal}>&times;</span>
            <h2>Group By Column</h2>
            <form onSubmit={handleGroupBySubmit} className="modal-body">
              <label className="form-label">Escolha a coluna para agrupar</label>
              <select
                className="select-styled"
                onChange={(e) => setSelectedColumn(e.target.value)}
                value={selectedColumn}
                required
              >
                <option value="">Selecione uma coluna</option>
                {columns.map((col, index) => (
                  <option key={index} value={col}>{col}</option>
                ))}
              </select>
              <div className="modal-actions">
                <button type="button" className="btn-ghost" onClick={closeGroupByModal}>Cancelar</button>
                <button type="submit" className="btn-cta">Group and Describe</button>
              </div>
            </form>
          </div>
        </div>
      )}

      <Modal
        isOpen={scatterPlotModalOpen}
        onClose={closeScatterPlotModal}
        title="Scatter Plot"
        contentClassName="null-modal"
      >
        <form onSubmit={(e) => {
          e.preventDefault();
          handleScatterPlotSubmit(scatterX, scatterY);
        }} className="modal-body">
          <div className="form-group">
            <label className="form-label">Selecione o eixo X</label>
            <select className="select-styled" onChange={(e) => setScatterX(e.target.value)} value={scatterX} required>
              <option value="">Select a column</option>
              {columns.map((col, index) => (
                <option key={index} value={col}>{col}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Selecione o eixo Y</label>
            <RadioGroup
              options={columns}
              selectedOption={scatterY}
              onChange={setScatterY}
            />
          </div>
          <div className="modal-actions">
            <button type="button" className="btn-ghost" onClick={closeScatterPlotModal}>Cancelar</button>
            <button type="submit" className="btn-cta" disabled={!scatterX || !scatterY}>Gerar Scatter Plot</button>
          </div>
        </form>
      </Modal>

      <Modal
        isOpen={boxplotModalOpen}
        onClose={closeBoxplotModal}
        title="Select Column for Boxplot"
      >
        <RadioGroup
          options={columns}
          selectedOption={selectedColumn}
          onChange={setSelectedColumn}
        />
        <button onClick={() => handleBoxplotSubmit(selectedColumn)} disabled={!selectedColumn}>
          Generate Boxplot
        </button>
      </Modal>

      <Modal
        isOpen={histogramModalOpen}
        onClose={closeHistogramModal}
        title="Histogramas"
      >
        <p>Gerar histogramas para todas as colunas numericas usando pandas DataFrame.hist.</p>
        <p>As configuracoes de bins e tamanho da figura sao ajustadas automaticamente pelo backend.</p>
        <button onClick={handleHistogramSubmit}>Gerar histogramas</button>
      </Modal>

      <Modal
        isOpen={categorizeModalOpen}
        onClose={closeCategorizeModal}
        title="Categorizar coluna"
        contentClassName="null-modal"
      >
        <form onSubmit={handleCategorizeSubmit} className="modal-body">
          <div className="form-group">
            <label className="form-label">Selecione a coluna</label>
            <select
              className="select-styled"
              value={categorizeColumn}
              onChange={(e) => setCategorizeColumn(e.target.value)}
              required
            >
              <option value="">Selecione</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Bins (opcional)</label>
            <input
              className="input-styled"
              type="number"
              min="2"
              max="20"
              placeholder="Auto"
              value={categorizeBins}
              onChange={(e) => setCategorizeBins(e.target.value)}
            />
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={categorizeAllowDecimals}
              onChange={(e) => setCategorizeAllowDecimals(e.target.checked)}
            />
            <span>Permitir numeros decimais nas categorias (desmarcado = inteiros)</span>
          </label>
          <div className="modal-actions">
            <button type="button" className="btn-ghost" onClick={closeCategorizeModal}>Cancelar</button>
            <button type="submit" className="btn-cta" disabled={!categorizeColumn}>Criar coluna categorizada</button>
          </div>
        </form>
      </Modal>

      <Modal
        isOpen={linearModalOpen}
        onClose={closeLinearModal}
        title="Regressao Linear Multipla"
        contentClassName="null-modal"
      >
        <form onSubmit={handleLinearRegressionSubmit} className="modal-body">
          <div className="form-group">
            <label className="form-label">Coluna target</label>
            <select
              className="select-styled"
              value={linearTargetColumn}
              onChange={(e) => setLinearTargetColumn(e.target.value)}
              required
            >
              <option value="">Selecione</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Test size (0 a 1)</label>
            <input
              className="input-styled"
              type="number"
              step="0.01"
              min="0.05"
              max="0.9"
              value={linearTestSize}
              onChange={(e) => setLinearTestSize(e.target.value)}
            />
          </div>
          <p className="muted">A coluna target sera removida do treino e as colunas numericas restantes serao usadas no modelo.</p>
          <div className="modal-actions">
            <button type="button" className="btn-ghost" onClick={closeLinearModal}>Cancelar</button>
            <button type="submit" className="btn-cta" disabled={!linearTargetColumn}>Rodar regressao</button>
          </div>
        </form>
      </Modal>

      {knnModalOpen && (
        <div className="modal">
          <div className="modal-content null-modal">
            <span className="close" onClick={closeKnnModal}>&times;</span>
            <h2>Configurar KNN</h2>
            <form onSubmit={handleKnnSubmit} className="modal-body">
              <div className="form-group">
                <label className="form-label">Max Neighbors</label>
                <input
                  className="input-styled"
                  type="number"
                  value={maxNeighbors}
                  onChange={(e) => setMaxNeighbors(e.target.value)}
                  min="3"
                />
              </div>
              <div className="form-group">
                <label className="form-label">Y Column</label>
                <select
                  className="select-styled"
                  onChange={(e) => setKnnY(e.target.value)}
                  value={knnY}
                  required
                >
                  <option value="">Select a column</option>
                  {columns.map((col, index) => (
                    <option key={index} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Imputer Strategy</label>
                <div>
                  <label>
                    <input type="radio" value="median" checked={imputerStrategy === 'median'} onChange={(e) => setImputerStrategy(e.target.value)} />
                    Median
                  </label>
                  <label>
                    <input type="radio" value="mean" checked={imputerStrategy === 'mean'} onChange={(e) => setImputerStrategy(e.target.value)} />
                    Mean
                  </label>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Scaler</label>
                <div>
                  <label>
                    <input type="radio" value="RobustScaler" checked={knnScaling === 'RobustScaler'} onChange={(e) => setKnnScaling(e.target.value)} />
                    RobustScaler
                  </label>
                  <label>
                    <input type="radio" value="StandardScaler" checked={knnScaling === 'StandardScaler'} onChange={(e) => setKnnScaling(e.target.value)} />
                    StandardScaler
                  </label>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Zero Missing Columns</label>
                <div className="null-checkbox-grid">
                  {columns.map(col => (
                    <label key={col} className="pill-checkbox">
                      <input 
                        type="checkbox" 
                        name={col}
                        checked={zeroMissingCols[col] || false} 
                        onChange={(e) => setZeroMissingCols(prev => ({...prev, [e.target.name]: e.target.checked}))} 
                      />
                      <span>{col}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Log Columns</label>
                <div className="null-checkbox-grid">
                  {columns.map(col => (
                    <label key={col} className="pill-checkbox">
                      <input 
                        type="checkbox" 
                        name={col}
                        checked={logCols[col] || false} 
                        onChange={(e) => setLogCols(prev => ({...prev, [e.target.name]: e.target.checked}))} 
                      />
                      <span>{col}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="modal-actions">
                <button type="button" className="btn-ghost" onClick={closeKnnModal}>Cancelar</button>
                <button type="submit" className="btn-cta">Executar KNN</button>
              </div>
            </form>
          </div>
        </div>
      )}

      <Modal
        isOpen={nullValuesModalOpen}
        onClose={closeNullValuesModal}
        title="Custom Null Values"
        contentClassName="null-modal"
      >
        <div className="null-modal-body">
          <p className="muted">Selecione as colunas e adicione marcadores de nulos customizados antes de checar/corrigir.</p>

          <div className="null-section">
            <div className="null-section__header">
              <h4>Colunas a verificar</h4>
              <label className="toggle">
                <input 
                  type="checkbox" 
                  onChange={(e) => {
                    const { checked } = e.target;
                    const allColumns = columns.reduce((acc, col) => {
                      acc[col] = checked;
                      return acc;
                    }, {});
                    setSelectedColumnsForNullCheck(allColumns);
                  }}
                />
                <span>Selecionar todas</span>
              </label>
            </div>
            <div className="null-checkbox-grid">
              {columns.map(col => (
                <label key={col} className="pill-checkbox">
                  <input 
                    type="checkbox" 
                    name={col}
                    checked={selectedColumnsForNullCheck[col] || false} 
                    onChange={handleCheckboxChange} 
                  />
                  <span>{col}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="null-section">
            <h4>Valores nulos customizados</h4>
            <div className="null-input-row">
              <input 
                type="text" 
                placeholder="Ex: 999, N/A, missing"
                value={currentCustomNullValue} 
                onChange={(e) => setCurrentCustomNullValue(e.target.value)} 
              />
              <div className="null-actions-inline">
                <button className="btn-primary" onClick={addCustomNullValue}>Adicionar</button>
                <button className="btn-ghost" onClick={clearCustomNullValues}>Limpar</button>
              </div>
            </div>
            <div className="null-chip-list">
              {customNullValues.length === 0 && <span className="muted">Nenhum valor customizado adicionado.</span>}
              {customNullValues.map((val, index) => (
                <span key={index} className="null-chip">{val}</span>
              ))}
            </div>
          </div>

          <div className="null-section">
            <h4>Como corrigir</h4>
            <div className="null-radio-group">
              <label className={`pill-radio ${fixStrategy === 'na' ? 'active' : ''}`}>
                <input
                  type="radio"
                  name="fix-strategy"
                  value="na"
                  checked={fixStrategy === 'na'}
                  onChange={(e) => setFixStrategy(e.target.value)}
                />
                <span>Trocar por NA</span>
              </label>
              <label className={`pill-radio ${fixStrategy === 'mean' ? 'active' : ''}`}>
                <input
                  type="radio"
                  name="fix-strategy"
                  value="mean"
                  checked={fixStrategy === 'mean'}
                  onChange={(e) => setFixStrategy(e.target.value)}
                />
                <span>Trocar pela media</span>
              </label>
              <label className={`pill-radio ${fixStrategy === 'median' ? 'active' : ''}`}>
                <input
                  type="radio"
                  name="fix-strategy"
                  value="median"
                  checked={fixStrategy === 'median'}
                  onChange={(e) => setFixStrategy(e.target.value)}
                />
                <span>Trocar pela mediana</span>
              </label>
            </div>
            <p className="muted">Somente uma opcao pode ser usada por vez. Para mean/median, apenas colunas numericas recebem o preenchimento.</p>
          </div>

          <div className="null-actions">
            <button className="btn-cta" onClick={fetchNullValues}>Checar valores nulos</button>
            <button className="btn-ghost" onClick={handleFixNullValues}>Corrigir nulos</button>
          </div>
        </div>
      </Modal>

      <Modal
        isOpen={reducedModalOpen}
        onClose={closeReducedModal}
        title="Salvar dataset reduzido"
        contentClassName="null-modal"
      >
        <div className="modal-body">
          <p className="muted">Selecione as colunas que deseja excluir antes de salvar.</p>
          <div className="null-section">
            <div className="null-section__header">
              <h4>Colunas para excluir</h4>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={columns.length > 0 && columns.every(col => selectedColumnsForReduced[col])}
                  onChange={(e) => handleReducedSelectAll(e.target.checked)}
                />
                <span>Selecionar todas</span>
              </label>
            </div>
            <div className="null-checkbox-grid">
              {columns.map(col => (
                <label key={col} className="pill-checkbox">
                  <input
                    type="checkbox"
                    name={col}
                    checked={selectedColumnsForReduced[col] || false}
                    onChange={handleReducedCheckboxChange}
                  />
                  <span>{col}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="null-actions">
            <button className="btn-ghost" onClick={closeReducedModal}>Cancelar</button>
            <button className="btn-cta" onClick={handleDownloadReducedDataset}>Gerar CSV reduzido</button>
          </div>
        </div>
      </Modal>

      <Modal
        isOpen={convertModalOpen}
        onClose={closeConvertModal}
        title="Converter str em num"
        contentClassName="null-modal"
      >
        <div className="modal-body">
          <p className="muted">Selecione as colunas (objetos) que deseja tentar converter para numerico (Int64).</p>
          <div className="null-section">
            <div className="null-section__header">
              <h4>Colunas para converter</h4>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={columns.length > 0 && columns.every(col => selectedColumnsForConvert[col])}
                  onChange={(e) => handleConvertSelectAll(e.target.checked)}
                />
                <span>Selecionar todas</span>
              </label>
            </div>
            <div className="null-checkbox-grid">
              {columns.map(col => (
                <label key={col} className="pill-checkbox">
                  <input
                    type="checkbox"
                    name={col}
                    checked={selectedColumnsForConvert[col] || false}
                    onChange={handleConvertCheckboxChange}
                  />
                  <span>{col}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="null-actions">
            <button className="btn-ghost" onClick={closeConvertModal}>Cancelar</button>
            <button className="btn-cta" onClick={handleFixDataset} disabled={fixingDataset}>
              {fixingDataset ? 'Convertendo...' : 'Converter selecionadas'}
            </button>
          </div>
        </div>
      </Modal>


      {fixModalOpen && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closeFixModal}>&times;</span>
            <h2>Dataset corrigido</h2>
            <p>{fixMessage || 'Dataset corrigido. Deseja salvar o novo arquivo?'}</p>
            <div className="modal-actions">
              <button onClick={handleDownloadFixedDataset}>Salvar novo dataset</button>
              <button onClick={closeFixModal}>Fechar</button>
            </div>
          </div>
        </div>
      )}

      {infoModalOpen && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closeInfoModal}>&times;</span>
            <h2>Dataframe Info</h2>
            {infoLoading && <div>Loading info...</div>}
            {infoError && <div style={{ color: 'red' }}>{infoError}</div>}
            {infoData && !infoLoading && !infoError && (
              <div className="info-content">
                <table className="grid">
                  <thead>
                    <tr>
                      <th>Key</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Rows</td><td>{infoData.rows}</td></tr>
                    <tr><td>Columns</td><td>{infoData.columns}</td></tr>
                    <tr><td>Index</td><td>{infoData.index}</td></tr>
                    <tr><td>Memory Usage (bytes)</td><td>{infoData.memoryUsageBytes}</td></tr>
                  </tbody>
                </table>

                <h4>Columns overview</h4>
                <div className="table-container" style={{ maxHeight: 260, overflow: 'auto' }}>
                  <table className="grid">
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Non-null</th>
                        <th>Dtype</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(infoData.dtypes || {}).map((col) => (
                        <tr key={col}>
                          <td>{col}</td>
                          <td>{infoData.nonNull ? infoData.nonNull[col] : '-'}</td>
                          <td>{infoData.dtypes[col]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <h4>Raw info()</h4>
                <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 180, overflow: 'auto' }}>
{infoData.infoText}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
