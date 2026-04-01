document.addEventListener('DOMContentLoaded', () => {
    const brandSelect = document.getElementById('brand');
    const modelSelect = document.getElementById('model');
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const predictedPriceText = document.getElementById('predicted-price');
    const fullPriceText = document.getElementById('full-price');
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
        resultContainer.classList.remove('error-glass');
        fullPriceText.textContent = '';

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
            
            if (data.status === 'success') {
                resultLabel.textContent = 'Estimated Market Value';
                predictedPriceText.className = 'result-value';
                
                // Format price in Lakhs
                const priceValue = data.predicted_price;
                const lakhsValue = (priceValue / 100000).toFixed(2);
                
                predictedPriceText.textContent = `${lakhsValue} Lakhs`;

                // Format full price as a secondary subtitle
                const formatter = new Intl.NumberFormat('en-LK', {
                    style: 'currency',
                    currency: 'LKR',
                    maximumFractionDigits: 0
                });
                fullPriceText.textContent = formatter.format(priceValue);
                
            } else {
                resultContainer.classList.add('error-glass');
                resultLabel.textContent = 'Error';
                predictedPriceText.className = 'result-value error';
                predictedPriceText.textContent = data.errors ? data.errors.join(', ') : 'Failed to predict';
                fullPriceText.textContent = '';
            }
        })
        .catch(err => {
            console.error('Prediction Error:', err);
            resultContainer.classList.remove('hidden');
            resultContainer.classList.add('error-glass');
            resultLabel.textContent = 'Error';
            predictedPriceText.className = 'result-value error';
            predictedPriceText.textContent = 'Failed to communicate with server.';
            fullPriceText.textContent = '';
        })
        .finally(() => {
            btnText.style.display = 'inline-block';
            spinner.style.display = 'none';
            submitBtn.disabled = false;
            
            // Auto scroll down slightly to show result
            setTimeout(() => {
                resultContainer.scrollIntoView({behavior: 'smooth', block: 'end'});
            }, 100);
        });
    });
});
