import React, { useState } from 'react';
import axios from 'axios';
import { FaFileCsv } from "react-icons/fa";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [selectedDays, setSelectedDays] = useState(1);

  const handleFileDrop = async (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'text/csv') {
      setFile(droppedFile);
      setUploadStatus("Subiendo archivo...");
      await uploadFile(droppedFile);
    } else {
      setUploadStatus("Por favor, suelta un archivo CSV.");
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload_csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploadStatus(response.data.message);
    } catch (error) {
      setUploadStatus("Error al subir el archivo.");
      console.error(error);
    }
  };

  const handleProcessData = async () => {
    try {
        const response = await axios.post(
            'http://localhost:5000/process_data',
            { days: selectedDays },
            { 
                responseType: 'json',
                headers: {
                    'Content-Type': 'application/json',
                }
            }
        );

        setMetrics(response.data.metrics);

        const base64Response = response.data.image;
        const binaryString = window.atob(base64Response);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const imageBlob = new Blob([bytes], { type: 'image/png' });
        const imageUrl = URL.createObjectURL(imageBlob);
        setImageUrl(imageUrl);
    } catch (error) {
        console.error("Error al procesar los datos:", error);
    }
};

  const handleGenerateImage = async () => {
    try {
      const response = await axios.get('http://localhost:5000/volatility', {
        responseType: 'blob',
      });
      const imageBlob = new Blob([response.data], { type: 'image/png' });
      const imageUrl = URL.createObjectURL(imageBlob);
      setImageUrl(imageUrl);
    } catch (error) {
      console.error("Error al generar la imagen:", error);
    }
  };

  const handleGenerateCiclos = async () => {
    try {
      const response = await axios.get('http://localhost:5000/ciclos', {
        responseType: 'blob',
      });
      const imageBlob = new Blob([response.data], { type: 'image/png' });
      const imageUrl = URL.createObjectURL(imageBlob);
      setImageUrl(imageUrl);
    } catch (error) {
      console.error("Error al generar la imagen:", error);
    }
  };

  return (
    <div className="p-4">
      <h2 className='text-5xl font-bold font-tertiary text-center mb-8'>Sección de predicción</h2>
      <p className='text-center mb-5 font-tertiary text-xl'>Arrastra un archivo del indice de su preferencia dentro del cuadro.</p>
      <p className='text-center mb-5 font-tertiary text-xl'>Seleccione cuantos días quiere predecir a partir de sus datos y aprete el botón <span className='text-green-700 font-bold'>Procesar Datos</span></p>
      <p className='text-center mb-5 font-tertiary text-xl'>Aprete el botón <span className='text-purple-500 font-bold'>Gráfico de volatilidad</span> para mostrar esta característica de su archivo</p>
      <p className='text-center mb-5 font-tertiary text-xl'>Aprete el botón <span className='text-red-500 font-bold'>Gráfico de ciclos</span> para mostrar los periodos bajistas y alcistas de su archivo</p>

      <div
        className="border-dashed border-4 border-green-700 p-6 rounded-md text-center w-[1500px] m-auto"
        onDrop={handleFileDrop}
        onDragOver={handleDragOver}
      >
        {file ? (
          <p>{file.name}</p>
        ) : (
          <div className='flex justify-center gap-4'>
            <div className='flex items-center'>
              <FaFileCsv size={32} className='text-green-700'></FaFileCsv>
            </div>
            <p>Arrastra y suelta un archivo CSV aquí</p>
          </div>
        )}
      </div>
      <div className='flex justify-between mt-20'>
        <div className='m-auto flex-1'>
            <div className="mt-4 flex flex-col">

                <div className='flex gap-4 mb-4 justify-center'> 
                  <h3 className='m-auto font-secondary'>Selecciona los días a predecir </h3>
                    <select 
                    name="days" 
                    id="pred_days" 
                    className="border border-blue-600"
                    value={selectedDays}
                    onChange={(e) => setSelectedDays(e.target.value)}
                    >
                        <option value={1}>1</option>
                        <option value={3}>3</option>
                        <option value={7}>7</option>
                        <option value={15}>15</option>
                    </select>
                    <button 
                    onClick={handleProcessData} 
                    className="bg-green-500 text-white px-4 py-2 rounded mr-2"
                    >
                    Procesar Datos
                    </button>
                </div>
                
                <button 
                onClick={handleGenerateImage} 
                className="bg-purple-500 text-white px-4 py-2 rounded m-4"
                >
                Gráfico de volatilidad
                </button>

                <button 
                onClick={handleGenerateCiclos} 
                className="bg-red-500 text-white px-4 py-2 rounded m-4"
                >
                Gráfico de ciclos
                </button>
            </div>

            {uploadStatus && (
                <p className="mt-5 text-center font-primary text-gray-700">{uploadStatus}</p>
            )}
        </div>
        <div className='flex flex-col flex-[3] justify-center items-center'>
            {imageUrl && (
                <div className="mt-4 w-[1000px]">
                    <img src={imageUrl} alt="Imagen generada" className="mt-2 border border-gray-400 rounded" />
                </div>
            )}
            {metrics && (
                <div className="mt-4 p-4 bg-white border border-gray-400 rounded">
                  <h3 className="text-lg font-semibold mb-2">Métricas de Predicción</h3>
                    <div className="grid grid-cols-3 gap-4">
                        <div>
                            <span className="font-medium">MAPE: </span>
                            <span>{Number(metrics.MAPE).toFixed(2)}%</span>
                        </div>
                        <div>
                            <span className="font-medium">MAE: </span>
                            <span>${Number(metrics.MAE).toFixed(2)}</span>
                        </div>
                        <div>
                            <span className="font-medium">RMSE: </span>
                            <span>${Number(metrics.RMSE).toFixed(2)}</span>
                        </div>
                    </div>
                </div>
              )}
        </div>
        </div>
    </div>
  );
};

export default FileUpload;
