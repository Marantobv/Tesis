// src/SP500Chart.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const Charts = () => {
    const [chartData, setChartData] = useState({
        labels: [],
        datasets: [
            {
                label: '',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
            },
        ],
    });

    useEffect(() => {
        const fetchData = async () => {
            const API_KEY = 'HWZMVGXZR20TW0J6';
            const symbol = 'SPY'; // SPY is an ETF that tracks the S&P500
            const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&apikey=${API_KEY}`;

            try {
                const response = await axios.get(url);
                const data = response.data['Time Series (Daily)'];
                console.log(response);
                if (data) {
                    const dates = Object.keys(data).reverse();
                    const closingPrices = dates.map(date => data[date]['4. close'] * 10);
                    
                    setChartData({
                        labels: dates,
                        datasets: [
                            {
                                label: 'S&P 500',
                                data: closingPrices,
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1,
                            },
                        ],
                    });
                } else {
                    console.error('No data found');
                }
            } catch (error) {
                console.error('Error fetching data from Alpha Vantage', error);
            }
        };

        fetchData();
    }, []);

    return (
        <div className='section' id='charts'>
            <div className='h-5/6 w-5/6 m-auto'>
                <h2 className='text-center font-bold text-4xl font-secondary'>S&P500 Chart</h2>
                <Line data={chartData} />
            </div>
        </div>
    );
};

export default Charts;
