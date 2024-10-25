import React, { useState } from 'react';
import axios from 'axios';

function NewsAndSentiment() {
  const [openPrice, setOpenPrice] = useState('');
  const [closePrice, setClosePrice] = useState('');
  const [message, setMessage] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const classifyNews = async () => {
    let allNews = [];
    let errorOccurred = false;

    for (let i = 1; i <= 5; i++) {
      try {
        const response = await axios.get('https://api.marketaux.com/v1/news/all', {
          params: {
            countries: 'us',
            language: 'en',
            symbols: 'TSLA,AMZN,MSFT,AAPL,NVDA,META,GOOGL,JPM,V',
            filter_entities: true,
            published_on: '2024-10-20',
            page: i,
            group_similar: false,
            api_token: 'aKYI2pUA2GE1pnbWPPpLjJVJTuMuh5MVDvfKG5MY'
          }
        });
        const newsData = response.data.data.map(news => ({
          title: news.title,
          description: news.description,
          date: '2024-10-20'
        }));
        allNews = [...allNews, ...newsData];
      } catch (error) {
        setMessage('Error al obtener las noticias de la página ' + i);
        errorOccurred = true;
        break;
      }
    }

    if (!errorOccurred) {
      try {
        await axios.post('http://127.0.0.1:5000/classify_news', allNews);
        setMessage('Noticias clasificadas y guardadas correctamente');
      } catch (error) {
        setMessage('Error al clasificar las noticias');
      }
    }
  };

  const addSentimentData = async () => {
    try {
      const data = {
        open_price: openPrice,
        close_price: closePrice
      };

      const response = await axios.post('http://127.0.0.1:5000/add_sentiment_data', data);
      setMessage('Datos agregados al CSV correctamente');
      setSentiment(response.data.average_sentiment_day);
    } catch (error) {
      setMessage('Error al agregar datos al CSV');
    }
  };

  const predictClosePrice = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get('http://127.0.0.1:5000/predict_close_price');
      setPrediction(response.data.predicted_close_price);
      setMessage('Predicción obtenida correctamente');
    } catch (error) {
      setMessage('Error al obtener la predicción');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='section' id='news-and-sentiment'>
      <div className='container mx-auto max-w-[1200px] text-center text-morado'>

        <div className='p-4'>
          <h2 className='text-5xl font-bold mb-4 font-tertiary'>Clasificación de Noticias y Sentimiento Promedio</h2>
          <button onClick={classifyNews} className='bg-blue-800 text-white px-4 py-2 rounded mb-4 font-bold'>
            Clasificar Noticias
          </button>
        </div>

        <div className='p-4'>
          <h2 className='text-5xl font-bold mb-4 font-tertiary text-morado'>Ingrese el precio de apertura y cierre de hoy</h2>
          <div className='mb-4'>
            <input
              type='text'
              placeholder='Precio de apertura'
              value={openPrice}
              onChange={(e) => setOpenPrice(e.target.value)}
              className='border p-2 mr-2'
            />
            <input
              type='text'
              placeholder='Precio de cierre'
              value={closePrice}
              onChange={(e) => setClosePrice(e.target.value)}
              className='border p-2 mr-2'
            />
          </div>
          <button onClick={addSentimentData} className='bg-green-800 text-white px-4 py-2 rounded font-bold'>
            Agregar datos al CSV
          </button>
        </div>

        {message && <p className='mt-4 text-lg'>{message}</p>}

        {sentiment !== null && (
          <p className='mt-4 text-lg'>Sentimiento promedio del día: {sentiment}</p>
        )}

        <div className='p-4'>
          <h2 className='text-5xl font-bold mb-4 font-tertiary text-morado'>Obtener predicción para mañana</h2>
          <button onClick={predictClosePrice} className='bg-red-800 text-white px-4 py-2 rounded font-bold'>
            Predecir precio de cierre
          </button>

          {isLoading && <div role="status">
    <svg aria-hidden="true" class="inline w-10 h-10 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
        <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
    </svg>
    <span class="sr-only">Cargando...</span>
</div>}

          {prediction !== null && (
            <p className='mt-4 text-lg'>Predicción del precio de cierre de mañana: {prediction}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default NewsAndSentiment;
