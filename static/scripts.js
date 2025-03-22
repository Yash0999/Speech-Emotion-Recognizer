document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const resultContainer = document.getElementById('result');
    const emotionDisplay = document.getElementById('emotionDisplay');

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) {
            emotionDisplay.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            emotionDisplay.innerHTML = '';
            data.predictions.forEach((prediction) => {
                const emotionCard = document.createElement('div');
                emotionCard.className = 'emotion-card';
                emotionCard.innerHTML = `
                    <span>${prediction.emotion} (${(prediction.probability * 100).toFixed(2)}%)</span>
                    <span class="emoji">${prediction.emoji}</span>
                `;
                emotionDisplay.appendChild(emotionCard);
            });
        }
        resultContainer.classList.remove('hidden');
    } catch (error) {
        emotionDisplay.innerHTML = `<p class="error">Failed to analyze emotion. Try again later.</p>`;
        resultContainer.classList.remove('hidden');
    }
});
