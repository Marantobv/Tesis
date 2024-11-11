import React, { useState } from 'react';
import axios from 'axios';
import { FaFileCsv } from "react-icons/fa";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageUrl2, setImageUrl2] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [selectedDays, setSelectedDays] = useState(1);

  // Manejar el archivo al soltar en el área de drag-and-drop
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

  // Evitar el comportamiento predeterminado al arrastrar archivos
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Subir el archivo CSV al backend
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

  // Procesar datos del CSV en el backend
  const handleProcessData = async () => {
    try {
        const response = await axios.post(
            'http://localhost:5000/process_data',
            { days: selectedDays }, // Envía el valor seleccionado al backend
            { responseType: 'blob' } // Blob para manejar la imagen
        );

        // Crear una URL para la imagen recibida
        const imageBlob = new Blob([response.data], { type: 'image/png' });
        const imageUrl = URL.createObjectURL(imageBlob);
        setImageUrl(imageUrl); // Guardar la URL en el estado
    } catch (error) {
        console.error("Error al procesar los datos:", error);
    }
};

  // Obtener imagen generada desde el backend
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
                  <h3 className='m-auto'>Selecciona los dias a predecir </h3>
                    <select 
                    name="days" 
                    id="pred_days" 
                    className="border border-blue-600"
                    value={selectedDays} // Valor controlado
                    onChange={(e) => setSelectedDays(e.target.value)} // Actualizar estado al cambiar el valor
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
        <div className='flex flex-[3] justify-center'>
            {imageUrl && (
                <div className="mt-4 w-[1000px]">
                    {/* <h3 className="text-xl font-bold text-center">Imagen Generada:</h3> */}
                    <img src={imageUrl} alt="Imagen generada" className="mt-2 border border-gray-400 rounded" />
                </div>
            )}
        </div>
        </div>
    </div>
  );
};

export default FileUpload;
