import React, { useState } from 'react';
import axios from 'axios';
import './Home.css'

function Home() {
    const [text, setText] = useState('');
    const [analyzerType, setAnalyzerType] = useState("basic");
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const analyzeClick = async () => {
        try {
            const response = await axios.post('/analysis/analyze/', {text:text, analyzer_type:analyzerType});
            setResult(response.data)
        } catch (error) {
            setError("Error Analyzing Text")
        } 
    };

    const renderResult = () => {
        if (result) {
            return (
                <div>
                    <h2>Result</h2>
                    <p>Sentiment: {result.sentiment}</p>
                    <p>Score: {result.score}</p>
                </div>
            );
        } else if (error) {
            return (
                <div>
                    <h2>{error}</h2>
                </div>
            );
        } else {
            return <p>Please input text and click "Analyze" to see the result.</p>;
        }
    };

    return (
        <div className='home'>
            <h1>Sentiment Analysis</h1>
            <div className='content'>
                <div className='select_container'>
                    <select value={analyzerType} onChange={(e) => setAnalyzerType(e.target.value)}>
                        <option value="basic">Basic Lexicon Analyzer</option>
                        <option value="advanced">Advanced Lexicon Analyzer</option>
                        <option value="logistic_regression">Logistic Regression Analyzer</option>
                        <option value="random_forest">Random Forest Analyzer</option>
                        <option value="BERT">BERT Analyzer</option>
                    </select>
                </div>
                <div className='textarea_container'>
                    <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder='Enter text to analyze.'/>
                    <button id='button' onClick={analyzeClick}>Analyze</button>
                    </div>
                <div className='result_container'>
                    {renderResult()}
                </div>
            </div>
        </div>
    );
}

export default Home;