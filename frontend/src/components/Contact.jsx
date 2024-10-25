import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { fade } from '../utils/variants';
import axios from 'axios';

function Contact() {
    const [responseMessage, setResponseMessage] = useState('');
    const [news, setNews] = useState([]);

    // Función para hacer la request al backend
    const handleGetNews = async () => {
        try {
            const response = await axios.get('http://127.0.0.1:5000/get_and_classify_news');
            setResponseMessage(response.data.message);
            setNews(response.data.classified_news);  // Almacenar las noticias clasificadas
        } catch (error) {
            console.error("Error al obtener las noticias", error);
            setResponseMessage('Error al obtener las noticias');
        }
    };

    return (
        <div className='section' id='contact'>
            <div className='container mx-auto max-w-[1200px]'>
                <h2>Obtener Noticias Clasificadas</h2>

                {/* Botón para realizar la request */}
                <button 
                    className='btn' 
                    onClick={handleGetNews}
                >
                    Obtener Noticias
                </button>

                {/* Mostrar el mensaje de respuesta */}
                {/* {responseMessage && <p>{responseMessage}</p>} */}

                {/* Mostrar las noticias clasificadas */}
                {/* {news.length > 0 && (
                    <div>
                        <h3>Noticias Clasificadas:</h3>
                        <ul>
                            {news.map((item, index) => (
                                <li key={index}>
                                    <strong>Título:</strong> {item.title} <br/>
                                    <strong>Descripción:</strong> {item.description} <br/>
                                    <strong>Sentimiento:</strong> {item.sentiment} <br/>
                                    <strong>Fecha:</strong> {item.date}
                                </li>
                            ))}
                        </ul>
                    </div>
                )} */}
            </div>
        </div>
    );
}

export default Contact;
