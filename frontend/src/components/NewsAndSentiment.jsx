import React, { useState } from 'react';
import axios from 'axios';

function NewsAndSentiment() {
  const [openPrice, setOpenPrice] = useState('');
  const [closePrice, setClosePrice] = useState('');
  const [message, setMessage] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [classifiedNews, setClassifiedNews] = useState([]);

  const todayDate = new Date().toISOString().split('T')[0];

  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);

  const tomorrowDate = tomorrow.toISOString().split('T')[0];

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
            published_on: todayDate,
            page: i,
            group_similar: false,
            api_token: 'vVKNV51Yl3p0LukGL7dLg5v2db6kTrGF3Xok8vUE'
          }
        });
        const newsData = response.data.data.map(news => ({
          title: news.title,
          description: news.description,
          date: todayDate 
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
        const response = await axios.post('http://127.0.0.1:5000/classify_news', allNews, {
          headers: { 'Content-Type': 'application/json' }
        });
        setClassifiedNews(response.data);  // Guardar el JSON de noticias clasificadas
        setMessage('Noticias clasificadas correctamente');
      } catch (error) {
        setMessage('Error al clasificar las noticias');
      }
    }
  };

  const addSentimentData = async () => {

    if (!openPrice || !closePrice) {
      setMessage('Por favor ingrese tanto el precio de apertura como el precio de cierre.');
      return; 
    }

    try {
      const data = {
        open_price: openPrice,
        close_price: closePrice
      };

      const response = await axios.post('http://127.0.0.1:5000/add_sentiment_data', data);
      setMessage('Datos agregados al CSV correctamente');
      setSentiment(response.data.average_sentiment_day);
    } catch (error) {
      console.log(error.response.data.message);
      
      setMessage(`${error.response.data.message}`);
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

  const getSentimentClass = (sentiment) => {
    switch (sentiment) {
      case 'negative':
        return 'text-red-700';
      case 'positive':
        return 'text-green-700';
      case 'neutral':
        return 'text-blue-700';
      default:
        return 'text-gray-700';
    }
  };

  return (
    <div className='section' id='news-and-sentiment'>
      <div className='container mx-auto max-w-[1200px] text-center text-morado'>

        {/* <div>
          <h2 className='font-bold text-4xl font-secondary'>Guía de usuario</h2>
          <p className='font-bold text-lg font-secondary text-black'>SP500: <span className='font-normal'>Es un indicador bursátil que mide el rendimiento de las 500 empresas más grandes y representativas que cotizan en las bolsas de Estados Unidos (principalmente NYSE y NASDAQ). Es ampliamente utilizado como referencia del desempeño general del mercado de valores y de la economía estadounidense.</span></p>
          <p className='font-bold text-lg font-secondary text-black'>Clasificación de noticias: <span className='font-normal'>Este botón se usa para obtener noticias del día de hoy y generar posteriormente el índice de sentimientos para el día de hoy. Se debe mostrar el mensaje de confirmación para continuar</span></p>
          <p className='font-bold text-lg font-secondary text-black'>Agregar datos al CSV: <span className='font-normal'>Este botón se usa para calcular el índice de sentimiento y almacenarlo junto a los datos de precios del SP500. En caso ya se haya generado los datos para hoy se muestra un mensaje de error. Para obtener los precios de cierre y apertura ingresar al siguiente link: <a target='_blank' className='text-blue-500' href='https://finance.yahoo.com/quote/%5ESPX/history/'>Precios SP500</a></span></p>
          <p className='font-bold text-lg font-secondary text-black'>Predecir precio de cierre: <span className='font-normal'>Este botón muestra la predicción para el día de mañana. </span></p>
        </div> */}

        <div className='p-4'>
          <h2 className='text-5xl font-bold mb-4 font-tertiary'>Clasificación de Noticias y Sentimiento Promedio</h2>
          <button onClick={classifyNews} className='bg-blue-800 text-white px-4 py-2 rounded mb-4 font-bold'>
            Clasificar Noticias
          </button>
          <p>{message}</p>
      <div className="news-list mt-4 grid grid-cols-2">
        {classifiedNews.map((news, index) => (
          <div key={index} className="border-2 border-yellow-400 m-4 rounded-xl p-2">
            <h3 className="font-bold text-lg font-tertiary">{news.title}</h3>
            <p className='font-secondary'>{news.description}</p>
            <p className='font-bold'>Fecha: <span className='font-normal'>{news.date}</span></p>
            <p className="font-bold">
              Sentimiento: <span className={`uppercase font-bold ${getSentimentClass(news.sentiment)}`}>
                {news.sentiment}
              </span>
            </p>
          </div>
        ))}
      </div>
        </div>

        <div className='p-4'>
          <h2 className='text-5xl font-bold mb-4 font-tertiary text-morado'>Ingrese el precio de apertura y cierre de hoy</h2>
          <div className='mb-4 flex justify-center gap-16'>
            <div className='flex flex-col text-left'>
            <label>Precio de apertura</label>
            <input
              type='text'
              placeholder='Ej. 5620.45'
              value={openPrice}
              onChange={(e) => setOpenPrice(e.target.value)}
              className='border p-2 mr-2 border-cyan-900 rounded'
            />
            </div>
            <div  className='flex flex-col text-left'>
            <label>Precio de cierre</label>
            <input
              type='text'
              placeholder='Ej. 5650.95'
              value={closePrice}
              onChange={(e) => setClosePrice(e.target.value)}
              className='border p-2 mr-2 border-cyan-900 rounded'
            />
            </div>
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
          <h2 className='text-5xl font-bold mb-4 font-tertiary text-morado'>Obtener predicción para mañana {tomorrowDate}</h2>
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
