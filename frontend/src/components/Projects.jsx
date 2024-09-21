import React, { useEffect, useState } from 'react';

function Projects() {
  const [news, setNews] = useState([]);

  // Obtiene noticias de dos páginas (3 noticias por página, total 6)
  useEffect(() => {
    const today = new Date().toISOString().split('T')[0]; // Fecha de hoy en formato "2024-09-21"
    const fetchNews = async () => {
      try {
        const response1 = await fetch(
          `https://api.marketaux.com/v1/news/all?countries=us&language=en&symbols=TSLA,AMZN,MSFT,AAPL,NVDA,META,GOOGL,JPM,V&filter_entities=true&published_on=${today}&page=1&group_similar=false&api_token=Y4tHtjZ8iP876QzQpMrVzsM08xnoK1HdEA2E5V39`
        );
        const response2 = await fetch(
          `https://api.marketaux.com/v1/news/all?countries=us&language=en&symbols=TSLA,AMZN,MSFT,AAPL,NVDA,META,GOOGL,JPM,V&filter_entities=true&published_on=${today}&page=2&group_similar=false&api_token=Y4tHtjZ8iP876QzQpMrVzsM08xnoK1HdEA2E5V39`
        );

        const data1 = await response1.json();
        const data2 = await response2.json();
        
        setNews([...data1.data, ...data2.data]); // Combina noticias de las dos páginas
      } catch (error) {
        console.error('Error fetching news:', error);
      }
    };

    fetchNews();
  }, []);

  return (
    <div className='section my-[50px]' id='projects'>
      <div className='container mx-auto max-w-[1200px] grid lg:grid-cols-3 grid-cols-2  gap-4'>
        {news.map((item, index) => (
          <div key={index} className='card border lg:p-4 p-2'>
            <img src={item.image_url} alt={item.title} className='mb-4 h-[200px] object-cover' />
            {/* <h2 className='lg:text-lg text-base font-bold mb-2'>{item.title}</h2> */}
            <a className='lg:text-lg text-base font-bold mb-2 cursor-pointer hover:underline text-morado font-secondary' target='_blank' href={item.url}>{item.title}</a>
            <p className='mb-2 font-secondary text-morado'>{item.description}</p>
            <p className='text-sm text-gray-500'>{item.source}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Projects;
