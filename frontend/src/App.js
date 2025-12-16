import React, { useState } from 'react';
import './App.css';

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
  const [scatterPlotImage, setScatterPlotImage] = useState('');
  const [boxplotImage, setBoxplotImage] = useState('');
  const [histogramImage, setHistogramImage] = useState('');
  const [knnModalOpen, setKnnModalOpen] = useState(false);
  const [nNeighbors, setNNeighbors] = useState(3);
  const [predictModalOpen, setPredictModalOpen] = useState(false);
  const [predictData, setPredictData] = useState({
    mean_radius: '',
    mean_area: '',
    mean_perimeter: '',
    mean_texture: '',
    mean_smoothness: ''
  });
  const [predictionResult, setPredictionResult] = useState(null);

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
  const openKnnModal = () => setKnnModalOpen(true);
  const closeKnnModal = () => setKnnModalOpen(false);
  const openPredictModal = () => setPredictModalOpen(true);
  const closePredictModal = () => {
    setPredictModalOpen(false);
    setPredictionResult(null);
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

  const fetchNullValues = async () => {
    try {
      const response = await fetch('http://localhost:5000/null-values');
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
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching null values:', error);
      alert('An error occurred while fetching null values.');
    }
  };

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

  const handleScatterPlotSubmit = async (plotSelection) => {
    if (!plotSelection) {
      alert("Please select a plot.");
      return;
    }
    const [x_col, y_col] = plotSelection.split('_vs_');
    try {
      const response = await fetch(`http://localhost:5000/scatterplot?x_col=${x_col}&y_col=${y_col}`);
      const data = await response.json();
      if (response.ok) {
        setScatterPlotImage(`data:image/png;base64,${data.image}`);
        setGridData([]);
        setGridTitle('');
        setHtmlContent('');
        closeScatterPlotModal();
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error fetching scatter plot data:', error);
      alert('An error occurred while fetching the scatter plot data.');
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


  const fetchTrainTest = async (n_neighbors) => {
   try {
     const response = await fetch(`http://localhost:5000/traintest?n_neighbors=${n_neighbors}`);
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
    fetchTrainTest(nNeighbors);
    closeKnnModal();
  };

  const handlePredictChange = (e) => {
    const { name, value } = e.target;
    setPredictData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handlePredictSubmit = async (e) => {
    e.preventDefault();
    if (!predictData.mean_radius || !predictData.mean_area || !predictData.mean_perimeter || !predictData.mean_texture || !predictData.mean_smoothness) {
      alert("Please fill in all fields.");
      return;
    }
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ...predictData, n_neighbors: nNeighbors }),
      });
      const data = await response.json();
      if (response.ok) {
        setPredictionResult(data.prediction);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error making prediction:', error);
      alert('An error occurred while making the prediction.');
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
        <button onClick={fetchNullValues} disabled={columns.length === 0}>Valores Nulos</button>
        <button onClick={openBoxplotModal} disabled={columns.length === 0}>Boxplot</button>
        <button onClick={openHistogramModal} disabled={columns.length === 0}>Histogram</button>
        <button onClick={openScatterPlotModal} disabled={columns.length === 0}>Scatter Plot</button>
        <button onClick={handleScatterPlot3dSubmit} disabled={columns.length === 0}>3D Scatter Plot</button>
        <button onClick={openKnnModal} disabled={columns.length === 0}>KNN (test & train)</button>
        <button onClick={openPredictModal} disabled={columns.length === 0}>Predict</button>
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

      {scatterPlotModalOpen && (
        <div className="modal">
            <div className="modal-content">
                <span className="close" onClick={closeScatterPlotModal}>&times;</span>
                <h2>Select Scatter Plot</h2>
                <div className="scatter-plot-options">
                    <button onClick={() => handleScatterPlotSubmit('mean_area_vs_mean_perimeter')}>
                        mean_area vs mean_perimeter
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_area_vs_mean_radius')}>
                        mean_area vs mean_radius
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_perimeter_vs_mean_radius')}>
                        mean_perimeter vs mean_radius
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_area_vs_mean_texture')}>
                        mean_area vs mean_texture
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_area_vs_mean_smoothness')}>
                        mean_area vs mean_smoothness
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_perimeter_vs_mean_texture')}>
                        mean_perimeter vs mean_texture
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_perimeter_vs_mean_smoothness')}>
                        mean_perimeter vs mean_smoothness
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_radius_vs_mean_texture')}>
                        mean_radius vs mean_texture
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_radius_vs_mean_smoothness')}>
                        mean_radius vs mean_smoothness
                    </button>
                    <button onClick={() => handleScatterPlotSubmit('mean_smoothness_vs_mean_texture')}>
                        mean_smoothness vs mean_texture
                    </button>
                </div>
            </div>
        </div>
      )}

      {boxplotModalOpen && (
       <div className="modal">
           <div className="modal-content">
               <span className="close" onClick={closeBoxplotModal}>&times;</span>
               <h2>Select Column for Boxplot</h2>
               <div className="boxplot-options">
                   {columns.map((col, index) => (
                       <button key={index} onClick={() => handleBoxplotSubmit(col)}>
                           {col}
                       </button>
                   ))}
               </div>
           </div>
       </div>
     )}

      {histogramModalOpen && (
       <div className="modal">
           <div className="modal-content">
               <span className="close" onClick={closeHistogramModal}>&times;</span>
               <h2>Select Column for Histogram</h2>
               <div className="histogram-options">
                   {columns.map((col, index) => (
                       <button key={index} onClick={() => handleHistogramSubmit(col)}>
                           {col}
                       </button>
                   ))}
               </div>
           </div>
       </div>
     )}

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
              <button type="submit">Run KNN</button>
            </form>
          </div>
        </div>
      )}

      {predictModalOpen && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closePredictModal}>&times;</span>
            <h2>Make a Prediction</h2>
            <form onSubmit={handlePredictSubmit} className="predict-form">
              <label>
                Mean Radius
                <input
                  type="number"
                  name="mean_radius"
                  value={predictData.mean_radius}
                  onChange={handlePredictChange}
                  placeholder="Mean Radius"
                  step="any"
                />
              </label>
              <label>
                Mean Area
                <input
                  type="number"
                  name="mean_area"
                  value={predictData.mean_area}
                  onChange={handlePredictChange}
                  placeholder="Mean Area"
                  step="any"
                />
              </label>
              <label>
                Mean Perimeter
                <input
                  type="number"
                  name="mean_perimeter"
                  value={predictData.mean_perimeter}
                  onChange={handlePredictChange}
                  placeholder="Mean Perimeter"
                  step="any"
                />
              </label>
              <label>
                Mean Texture
                <input
                  type="number"
                  name="mean_texture"
                  value={predictData.mean_texture}
                  onChange={handlePredictChange}
                  placeholder="Mean Texture"
                  step="any"
                />
              </label>
              <label>
                Mean Smoothness
                <input
                  type="number"
                  name="mean_smoothness"
                  value={predictData.mean_smoothness}
                  onChange={handlePredictChange}
                  placeholder="Mean Smoothness"
                  step="any"
                />
              </label>
              <button type="submit">Predict</button>
            </form>
            {predictionResult !== null && (
              <div>
                <h3>Prediction Result:</h3>
                <p>{predictionResult === 0 ? "No Cancer (0)" : "Cancer (1)"}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
