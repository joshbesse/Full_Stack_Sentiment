import React, { useEffect, useState } from 'react';
import axios from 'axios';

function History() {
    const [history, setHistory] = useState([]);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await axios.get('/analysis/history/');
                setHistory(response.data)
            } catch (error) {
                setError("Error Fetching History")
            }
        };

        fetchHistory();
    }, []);

    return (
        <div className='history'>
            <h1>Analysis History</h1>
            <ul>
                {history.map((item, index) => (
                    <li key={index}>
                        <p>Text: {item.text}</p>
                        <p>Sentiment: {item.sentiment}</p>
                        <p>Score: {item.score}</p>
                    </li>
                ))}
            </ul>
        </div>
    );
}

export default History