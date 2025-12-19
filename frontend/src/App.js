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
  const [selectedColumn, setSelectedColumn] = useState('');
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterPlotImage, setScatterPlotImage] = useState('');
  const [boxplotImage, setBoxplotImage] = useState('');
  const [histogramImage, setHistogramImage] = useState('');
  const [knnModalOpen, setKnnModalOpen] = useState(false);
  const [nNeighbors, setNNeighbors] = useState(3);
  const [knnY, setKnnY] = useState('');
  const [predictAlso, setPredictAlso] = useState(false);
  const [predictValues, setPredictValues] = useState({});
  const [nullValuesModalOpen, setNullValuesModalOpen] = useState(false);
  const [customNullValues, setCustomNullValues] = useState([]);
  const [currentCustomNullValue, setCurrentCustomNullValue] = useState('');
  const [selectedColumnsForNullCheck, setSelectedColumnsForNullCheck] = useState({});

  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [infoData, setInfoData] = useState(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [infoError, setInfoError] = useState(null);
  const [fixModalOpen, setFixModalOpen] = useState(false);
  const [fixingDataset, setFixingDataset] = useState(false);
  const [fixMessage, setFixMessage] = useState('');

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
        setHtmlContent('');
        setInfoData(null);
        setInfoError(null);
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
  const openKnnModal = () => {
    setKnnModalOpen(true);
  };
  const closeKnnModal = () => setKnnModalOpen(false);

  const closeInfoModal = () => {
    setInfoModalOpen(false);
    setInfoLoading(false);
  };

  const fetchHead = async () => {
    try {
      const response = await fetch('http://localhost:5000/head');
      const data = await response.json();
      if (response.ok) {
        setGridData(data);
        setGridTitle('DataFrame Head:');
        setScatterPlotImage('');
        setBoxplotImage('');
        setHistogramImage('');
        setHtmlContent('');
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
        setBoxplotImage('');
        setHistogramImage('');
        setHtmlContent('');
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
        setHtmlContent('');
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
        body: JSON.stringify({ custom_null_values: customNullValues, columns_to_fix: selectedColumns }),
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
    setFixingDataset(true);
    try {
      const response = await fetch('http://localhost:5000/fix-dataset', { method: 'POST' });
      const data = await response.json();
      if (response.ok) {
        setColumns(data.columns || columns);
        setFixMessage(data.message || 'Dataset fixed successfully.');
        setFixModalOpen(true);
        setGridData([]);
        setGridTitle('');
        setScatterPlotImage('');
        setBoxplotImage('');
        setHistogramImage('');
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
        setHtmlContent('');
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
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setScatterPlotImage('');
        closeBoxplotModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching boxplot data:', error);
      alert('An error occurred while fetching the boxplot data.');
    }
  };

  const handleHistogramSubmit = async (column) => {
    if (!column) {
      alert("Please select a column.");
      return;
    }
    try {
      const response = await fetch(`http://localhost:5000/histogram?column=${column}`);
      const data = await response.json();
      if (response.ok) {
        setHistogramImage(`data:image/png;base64,${data.image}`);
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setScatterPlotImage('');
        setBoxplotImage('');
        closeHistogramModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching histogram data:', error);
      alert('An error occurred while fetching the histogram data.');
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
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        setBoxplotImage('');
        setHistogramImage('');
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
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching 3D scatter plot data:', error);
      alert('An error occurred while fetching the 3D scatter plot data.');
    }
  };


  const fetchTrainTest = async (n_neighbors, y_col, predict, predict_values) => {
    try {
      let url = `http://localhost:5000/traintest?n_neighbors=${n_neighbors}&y_col=${y_col}&predict=${predict}`;
      
      const options = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ predict_values: predict_values || {} })
      };

      const response = await fetch(url, options);
      const data = await response.json();
      
      if (response.ok) {
        setHtmlContent(data.html);
        setGridData([]);
        setGridTitle('');
        setScatterPlotImage('');
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

    if (predictAlso) {
      const requiredColumns = columns.filter(c => c !== knnY);
      const allFieldsFilled = requiredColumns.every(col => predictValues[col] && predictValues[col].trim() !== '');
      
      if (!allFieldsFilled) {
        alert("Please fill in all prediction values.");
        return;
      }
    }

    fetchTrainTest(nNeighbors, knnY, predictAlso, predictValues);
    closeKnnModal();
  };

  const handlePredictChange = (column, value) => {
    setPredictValues(prev => ({ ...prev, [column]: value }));
  };



  const handleDownloadReducedDataset = async () => {
    if (columns.length === 0) {
      alert("Please upload a dataset first.");
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
      const response = await fetch(`http://localhost:5000/download-reduced-dataset?filename=${encodeURIComponent(finalName)}`);
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
        <button onClick={handleFixDataset} disabled={columns.length === 0 || fixingDataset}>
          {fixingDataset ? 'Fixing...' : 'Fix dataset'}
        </button>
        <button onClick={handleDownloadReducedDataset} disabled={columns.length === 0}>Salvar reduzido</button>
        <button onClick={openBoxplotModal} disabled={columns.length === 0}>Boxplot</button>
        <button onClick={openHistogramModal} disabled={columns.length === 0}>Histogram</button>
        <button onClick={openScatterPlotModal} disabled={columns.length === 0}>Scatter Plot</button>
        <button onClick={handleScatterPlot3dSubmit} disabled={true}>3D Scatter Plot</button>
        <button onClick={openKnnModal} disabled={columns.length === 0}>KNN (test & train)</button>
      </div>
      <div className="content">
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
            <img src={scatterPlotImage} alt="Scatter Plot" />
          </div>
        )}
       {boxplotImage && (
         <div>
           <h2>Boxplot:</h2>
           <img src={boxplotImage} alt="Boxplot" />
         </div>
       )}
       {histogramImage && (
         <div>
           <h2>Histogram:</h2>
           <img src={histogramImage} alt="Histogram" />
         </div>
       )}
        {htmlContent && (
           <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
        )}
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
            <form onSubmit={handleGroupBySubmit}>
              <select onChange={(e) => setSelectedColumn(e.target.value)} value={selectedColumn}>
                <option value="">Select a column</option>
                {columns.map((col, index) => (
                  <option key={index} value={col}>{col}</option>
                ))}
              </select>
              <button type="submit">Group and Describe</button>
            </form>
          </div>
        </div>
      )}

      <Modal
        isOpen={scatterPlotModalOpen}
        onClose={closeScatterPlotModal}
        title="Scatter Plot"
      >
        <form onSubmit={(e) => {
          e.preventDefault();
          handleScatterPlotSubmit(scatterX, scatterY);
        }}>
          <div className="form-group">
            <label>Select X-axis:</label>
            <select onChange={(e) => setScatterX(e.target.value)} value={scatterX} required>
              <option value="">Select a column</option>
              {columns.map((col, index) => (
                <option key={index} value={col}>{col}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Select Y-axis:</label>
            <RadioGroup
              options={columns}
              selectedOption={scatterY}
              onChange={setScatterY}
            />
          </div>
          <button type="submit" disabled={!scatterX || !scatterY}>Generate Scatter Plot</button>
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
        title="Select Column for Histogram"
      >
        <RadioGroup
          options={columns}
          selectedOption={selectedColumn}
          onChange={setSelectedColumn}
        />
        <button onClick={() => handleHistogramSubmit(selectedColumn)} disabled={!selectedColumn}>
          Generate Histogram
        </button>
      </Modal>

      {knnModalOpen && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closeKnnModal}>&times;</span>
            <h2>Set K-Neighbors</h2>
            <form onSubmit={handleKnnSubmit} className="knn-form">
              <label>
                Number of Neighbors (n_neighbors):
                <input
                  type="number"
                  value={nNeighbors}
                  onChange={(e) => setNNeighbors(e.target.value)}
                  min="1"
                />
              </label>
              <label>
                Y Column:
                <select onChange={(e) => setKnnY(e.target.value)} value={knnY}>
                  <option value="">Select a column</option>
                  {columns.map((col, index) => (
                    <option key={index} value={col}>{col}</option>
                  ))}
                </select>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={predictAlso}
                  onChange={(e) => setPredictAlso(e.target.checked)}
                />
                Predict also
              </label>

              {predictAlso && (
                <div className="predict-values">
                  <h3>Enter values for prediction:</h3>
                  {columns.filter(c => c !== knnY).map(col => (
                    <div key={col} className="form-group">
                      <label>{col}:</label>
                      <input
                        type="number"
                        step="any"
                        value={predictValues[col] || ''}
                        onChange={(e) => handlePredictChange(col, e.target.value)}
                      />
                    </div>
                  ))}
                </div>
              )}
              
              <button type="submit">Run KNN</button>
            </form>
          </div>
        </div>
      )}

      <Modal
        isOpen={nullValuesModalOpen}
        onClose={closeNullValuesModal}
        title="Custom Null Values"
      >
        <div>
          <p>Columns to be checked:</p>
          <div className="form-group">
            <label>
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
              Select All
            </label>
          </div>
          <div className="checkbox-group">
            {columns.map(col => (
              <label key={col}>
                <input 
                  type="checkbox" 
                  name={col}
                  checked={selectedColumnsForNullCheck[col] || false} 
                  onChange={handleCheckboxChange} 
                />
                {col}
              </label>
            ))}
          </div>
          <div className="form-group">
            <label>Custom Null Value:</label>
            <input 
              type="text" 
              value={currentCustomNullValue} 
              onChange={(e) => setCurrentCustomNullValue(e.target.value)} 
            />
            <button onClick={addCustomNullValue}>Add</button>
            <button onClick={clearCustomNullValues}>Clear</button>
          </div>
          <div>
            <p>Custom Null Values List:</p>
            <ul>
              {customNullValues.map((val, index) => <li key={index}>{val}</li>)}
            </ul>
          </div>
          <button onClick={fetchNullValues}>Check Null Values</button>
          <span style={{ marginRight: '10px' }}></span>
          <button onClick={handleFixNullValues}>Corrigir Nulos</button>
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
