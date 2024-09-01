import React from 'react';

function Description() {
    return (
        <div className='description'>
            <h1>Sentiment Analysis Methods</h1>
            <h2>What is Sentiment Analysis?</h2>
            <p>Sentiment analysis is a field of natural language processing, which is the ability for computers to process human-created text. Sentiment analysis involves analyzing, understanding, and interpreting text to determine if the emotional tone of the message is positive, negative, or neutral.</p>
            <h2>Basic Lexicon Analyzer</h2>
            <p>The input text is broken down into individual words. Those words are then checked against lists of predefined positive and negative words. If the word is part of the positive words list, the sentiment score is incremented by 1. If the word is part of the negative words list, the sentiment score is decremented by 1. Based on the final sentiment score, a sentiment of positive, negative, or neutral is assigned.</p>
            <h2>Advanced Lexicon Analyzer</h2>
            <p>The advanced analyzer works similarly to the basic analyzer but with added features, specifically negation, intensifying, and diminishing handling. It has additional lists of negation words (example: didn't), intensifier words (example: very), and diminisher words (example: somewhat). The advanced analyzer checks the previous word to see if it part of the negation, intensifier, or diminisher lists. If it is a negation word, the score for the current word is multiplied by -1. If it is an intensifier word, the score for the current word is multiplied by 1.5. If it is a diminisher word, the score for the current word is multiplied by 0.5. </p>
        </div>
    );
}

export default Description