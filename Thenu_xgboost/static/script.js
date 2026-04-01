document.addEventListener('DOMContentLoaded', () => {
    const brandSelect = document.getElementById('brand');
    const modelSelect = document.getElementById('model');
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const predictedPriceText = document.getElementById('predicted-price');
    const resultLabel = document.querySelector('.result-label');

    let modelsData = {};

    // Fetch initial options
    fetch('/api/metadata')
        .then(res => res.json())
        .then(data => {
            modelsData = data.models_by_brand;
            
            // Populate brands
            data.brands.forEach(brand => {
                const opt = document.createElement('option');
                opt.value = brand;
                opt.textContent = brand;
                brandSelect.appendChild(opt);
            });
        })
        .catch(err => console.error('Error fetching metadata:', err));

    brandSelect.addEventListener('change', (e) => {
        const brand = e.target.value;
        const models = modelsData[brand] || [];
        
        // Reset model select
        modelSelect.innerHTML = '<option value="" disabled selected>Select Model</option>';
        modelSelect.disabled = models.length === 0;

        models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model;
            modelSelect.appendChild(opt);
        });
    });

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const btnText = submitBtn.querySelector('.btn-text');
        const spinner = document.getElementById('loading-spinner');
        
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');

        const formData = new FormData(form);
        const inputDict = {};
        
        formData.forEach((value, key) => {
            if (['Engine (cc)', 'Millage(KM)', 'Car_Age'].includes(key)) {
                inputDict[key] = Number(value);
            } else {
                inputDict[key] = value;
            }
        });

        // Add checkboxes since FormData only tracks checked ones
        ['AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW'].forEach(key => {
            inputDict[key] = form.elements[key].checked ? 1 : 0;
        });

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputDict)
        })
        .then(res => res.json())
        .then(data => {
            resultContainer.classList.remove('hidden');
            resultContainer.style.height = 'auto'; // allow expansion
            resultContainer.style.padding = '30px'; 
            resultContainer.style.border = '1px solid var(--primary-color)';
            
            if (data.status === 'success') {
                resultLabel.textContent = 'Estimated Market Value';
                predictedPriceText.className = 'result-value';
                
                // Format price in local currency
                const formatter = new Intl.NumberFormat('en-LK', {
                    style: 'currency',
                    currency: 'LKR',
                    maximumFractionDigits: 0
                });
                
                predictedPriceText.textContent = formatter.format(data.predicted_price);
            } else {
                resultLabel.textContent = 'Error';
                predictedPriceText.className = 'result-value error';
                predictedPriceText.textContent = data.errors ? data.errors.join(', ') : 'Failed to predict';
            }
        })
        .catch(err => {
            console.error('Prediction Error:', err);
            resultContainer.classList.remove('hidden');
            resultLabel.textContent = 'Error';
            predictedPriceText.className = 'result-value error';
            predictedPriceText.textContent = 'Failed to communicate with server.';
        })
        .finally(() => {
            btnText.style.display = 'inline-block';
            spinner.style.display = 'none';
            submitBtn.disabled = false;
        });
    });
});
