import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Car, Calculator, CheckCircle, AlertCircle, Sparkles, Activity, Search, ShieldCheck, Zap, BarChart3, TrendingUp, ScatterChart, PieChart } from 'lucide-react';
import axios from 'axios';
import './App.css';

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const popIn = {
  hidden: { scale: 0.9, opacity: 0 },
  visible: { scale: 1, opacity: 1 }
};

const TOP_BRANDS = ["TOYOTA", "HONDA", "NISSAN", "BMW", "AUDI", "MERCEDES-BENZ", "KIA", "SUZUKI"];

export default function App() {
  const [metadata, setMetadata] = useState({ brands: [], models_by_brand: {} });
  const [loadingMeta, setLoadingMeta] = useState(true);
  
  const [formData, setFormData] = useState({
    Brand: '',
    Model: '',
    Gear: 'Automatic',
    'Fuel Type': 'Petrol',
    'Engine (cc)': 1000,
    'Millage(KM)': 50000,
    Car_Age: 5,
    Condition: 'USED',
    'AIR CONDITION': 1,
    'POWER STEERING': 1,
    'POWER MIRROR': 1,
    'POWER WINDOW': 1,
    Leasing: '0'
  });

  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('/api/metadata')
      .then(res => {
        setMetadata(res.data);
        setLoadingMeta(false);
      })
      .catch(err => {
        console.error("Failed to fetch metadata", err);
        setLoadingMeta(false);
      });
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    setFormData(prev => {
      const updated = { ...prev };
      
      if (type === 'checkbox') {
        updated[name] = checked ? 1 : 0;
      } else if (['Engine (cc)', 'Millage(KM)', 'Car_Age'].includes(name)) {
        updated[name] = Number(value);
      } else {
        updated[name] = value;
      }

      // Reset model when brand changes
      if (name === 'Brand') updated.Model = '';
      
      return updated;
    });
  };

  const selectBrand = (brand) => {
    setFormData(prev => ({ ...prev, Brand: brand, Model: '' }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!formData.Brand) {
      setError("Please select a Make before predicting.");
      return;
    }
    setIsLoading(true);
    setPrediction(null);
    setError(null);

    try {
      const res = await axios.post('/api/predict', formData);
      if (res.data.status === 'success') {
        setPrediction(res.data.predicted_price);
      } else {
        setError(res.data.errors ? res.data.errors.join(', ') : "Failed to predict");
      }
    } catch (err) {
      setError("Failed to reach server.");
    } finally {
      setIsLoading(false);
    }
  };

  const availableModels = formData.Brand ? (metadata.models_by_brand[formData.Brand] || []) : [];
  const otherBrands = metadata.brands.filter(b => !TOP_BRANDS.includes(b));

  return (
    <>
      <div className="bg-mesh" />
      
      {/* Navbar */}
      <nav style={{ position: 'fixed', top: 0, width: '100%', height: '80px', background: 'rgba(8, 15, 30, 0.85)', backdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(255,255,255,0.05)', zIndex: 100 }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px'}}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ background: 'linear-gradient(135deg, #0ea5e9, #3b82f6)', padding: 8, borderRadius: 12 }}>
               <Car size={26} color="#ffffff" />
            </div>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'white', letterSpacing: '-0.5px' }}>
              Autolytica
            </h1>
          </div>
          <div style={{ display: 'flex', gap: 24, fontWeight: 500, fontSize: '0.95rem' }}>
             <a href="#predictor" style={{ color: 'white', textDecoration: 'none' }}>Valuation</a>
             <a href="#analytics" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Model Analytics</a>
          </div>
        </div>
      </nav>

      <main style={{ paddingTop: 140, paddingBottom: 100, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        
        {/* Hero */}
        <motion.div 
          initial="hidden" animate="visible" variants={fadeIn} transition={{duration: 0.6}}
          style={{ textAlign: 'center', maxWidth: 900, marginBottom: 80, padding: '0 24px' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'linear-gradient(90deg, rgba(14, 165, 233, 0.1), rgba(59, 130, 246, 0.1))', border: '1px solid rgba(14, 165, 233, 0.2)', padding: '8px 20px', borderRadius: 30, color: '#0ea5e9', fontSize: '0.9rem', fontWeight: 600, marginBottom: 32}}>
            <Sparkles size={16} /> Empowered by XGBoost Machine Learning
          </div>
          <h2 style={{ fontSize: '4.5rem', fontWeight: 800, lineHeight: 1.1, marginBottom: 24, letterSpacing: '-1.5px', textShadow: '0 10px 30px rgba(0,0,0,0.5)'}}>
            Precision Vehicle Valuation.<br/>
            <span style={{background: 'linear-gradient(to right, #94a3b8, #475569)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'}}>Data Driven.</span>
          </h2>
          <p style={{ fontSize: '1.3rem', color: 'var(--text-secondary)', maxWidth: 600, margin: '0 auto'}}>
            Select your brand, define the specs, and let our ML model extrapolate extreme market precision in milliseconds.
          </p>
        </motion.div>

        {/* Form Container */}
        <motion.div 
          id="predictor"
          initial="hidden" animate="visible" variants={fadeIn} transition={{duration: 0.6, delay: 0.2}}
          className="glass-panel" style={{ width: '100%', maxWidth: 1000, padding: '48px 40px', marginBottom: 80 }}>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 40 }}>
            <Search size={28} color="#0ea5e9" />
            <h3 style={{ fontSize: '2rem', fontWeight: 700 }}>Find Market Value</h3>
          </div>

          <form onSubmit={handlePredict} style={{ display: 'flex', flexDirection: 'column', gap: 40 }}>
            
            {/* Custom Brand Selector Grid */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <label style={{fontSize: '0.9rem', color: 'var(--primary-color)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1}}>1. Select Make</label>
              
              {loadingMeta ? (
                 <div style={{ color: 'var(--text-secondary)' }}>Loading brands...</div>
              ) : (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 16 }}>
                    {TOP_BRANDS.map(brand => (
                      <motion.div 
                        whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                        key={brand} 
                        onClick={() => selectBrand(brand)}
                        style={{
                          background: formData.Brand === brand ? 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))' : 'rgba(255,255,255,0.03)',
                          border: `1px solid ${formData.Brand === brand ? 'transparent' : 'rgba(255,255,255,0.1)'}`,
                          borderRadius: 16, padding: '24px 12px', textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s',
                          boxShadow: formData.Brand === brand ? '0 10px 20px rgba(14, 165, 233, 0.3)' : 'none'
                        }}
                      >
                        <span style={{ fontWeight: 700, fontSize: '0.9rem', color: formData.Brand === brand ? 'white' : 'var(--text-secondary)' }}>
                          {brand}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                  
                  {/* Other Brands Dropdown */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 8 }}>
                     <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Or select other:</span>
                     <select 
                       value={TOP_BRANDS.includes(formData.Brand) ? "" : formData.Brand} 
                       onChange={(e) => selectBrand(e.target.value)} 
                       className="input-base" style={{ width: 240 }}
                     >
                       <option value="" disabled>Other Brands</option>
                       {otherBrands.map(b => <option key={b} value={b}>{b}</option>)}
                     </select>
                  </div>
                </>
              )}
            </div>

            <div style={{ height: 1, width: '100%', background: 'linear-gradient(90deg, transparent, var(--border), transparent)' }} />

            {/* Core Specs Grid */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <label style={{fontSize: '0.9rem', color: 'var(--primary-color)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1}}>2. Vehicle Details</label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 24 }}>
                
                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Model</label>
                  <select name="Model" value={formData.Model} onChange={handleChange} className="input-base" required disabled={!formData.Brand}>
                    <option value="" disabled>{formData.Brand ? 'Select Model' : 'Select Make First'}</option>
                    {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Registration Age (Years)</label>
                  <input type="number" name="Car_Age" min="0" max="50" value={formData.Car_Age} onChange={handleChange} className="input-base" required />
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Mileage (KM)</label>
                  <input type="number" name="Millage(KM)" min="0" max="1000000" step="1000" value={formData['Millage(KM)']} onChange={handleChange} className="input-base" required />
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Engine Capacity (cc)</label>
                  <input type="number" name="Engine (cc)" min="500" max="10000" step="100" value={formData['Engine (cc)']} onChange={handleChange} className="input-base" required />
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Transmission</label>
                  <select name="Gear" value={formData.Gear} onChange={handleChange} className="input-base">
                    <option value="Automatic">Automatic</option>
                    <option value="Manual">Manual</option>
                  </select>
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Fuel Type</label>
                  <select name="Fuel Type" value={formData['Fuel Type']} onChange={handleChange} className="input-base">
                    <option value="Petrol">Petrol</option>
                    <option value="Diesel">Diesel</option>
                    <option value="Hybrid">Hybrid</option>
                    <option value="Electric">Electric</option>
                  </select>
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Market Condition</label>
                  <select name="Condition" value={formData.Condition} onChange={handleChange} className="input-base">
                    <option value="NEW">Brand New</option>
                    <option value="USED">Pre-Owned</option>
                  </select>
                </div>

                <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Active Leasing</label>
                  <select name="Leasing" value={formData.Leasing} onChange={handleChange} className="input-base">
                    <option value="0">Unencumbered (None)</option>
                    <option value="Ongoing Lease">Ongoing Lease</option>
                  </select>
                </div>
              </div>
            </div>

            <div style={{ height: 1, width: '100%', background: 'linear-gradient(90deg, transparent, var(--border), transparent)' }} />

            {/* Extra Options */}
            <div>
               <label style={{fontSize: '0.9rem', color: 'var(--primary-color)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, display: 'block', marginBottom: 20}}>3. Add-on Features</label>
               <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 20 }}>
                  {[
                    { label: 'A/C System', name: 'AIR CONDITION' },
                    { label: 'Power Steering', name: 'POWER STEERING' },
                    { label: 'Power Mirror', name: 'POWER MIRROR' },
                    { label: 'Power Windows', name: 'POWER WINDOW' }
                  ].map(feat => (
                    <label key={feat.name} style={{ 
                        display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer', fontSize: '0.95rem',
                        background: 'rgba(255,255,255,0.02)', padding: '12px 16px', borderRadius: 12, border: '1px solid var(--border)',
                        transition: 'all 0.2s', boxShadow: formData[feat.name] === 1 ? 'inset 0 0 0 1px var(--primary-color)' : 'none'
                      }}>
                      <div style={{ position: 'relative', width: 24, height: 24, background: formData[feat.name] ? 'var(--primary-color)' : 'rgba(255,255,255,0.1)', borderRadius: 6, transition: '0.2s', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                         <input type="checkbox" name={feat.name} checked={formData[feat.name] === 1} onChange={handleChange} style={{ opacity: 0, position: 'absolute', inset: 0, cursor: 'pointer' }} />
                         {formData[feat.name] === 1 && <CheckCircle size={16} color="white" />}
                      </div>
                      <span style={{ fontWeight: formData[feat.name] ? 600 : 400, color: formData[feat.name] ? 'white' : 'var(--text-secondary)'}}>{feat.label}</span>
                    </label>
                  ))}
               </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', marginTop: 16 }}>
              <button type="submit" className="btn-glow" disabled={isLoading || !formData.Brand} style={{ maxWidth: 400, height: 64, fontSize: '1.15rem' }}>
                {isLoading ? (
                   <>
                     <div style={{ width: 24, height: 24, borderRadius: '50%', border: '3px solid rgba(255,255,255,0.3)', borderTopColor: 'white', animation: 'spin 1s linear infinite' }} />
                     Evaluating Deep Model...
                   </>
                ) : (
                  <>
                    <Activity size={24} /> Calculate Exact Value
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Result Animation */}
          <AnimatePresence>
             {prediction && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }} 
                  animate={{ opacity: 1, scale: 1, marginTop: 40 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  style={{ overflow: 'hidden' }}
                >
                   <div style={{ 
                     background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05))', 
                     border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: 24, padding: '48px 32px', textAlign: 'center',
                     boxShadow: '0 20px 40px rgba(16, 185, 129, 0.1)'
                   }}>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: '#34d399', fontWeight: 700, marginBottom: 16, textTransform: 'uppercase', letterSpacing: 1.5, fontSize: '0.9rem'}}>
                        <CheckCircle size={20} /> Market Value Verified
                      </div>
                      <h2 style={{ fontSize: '4.5rem', fontWeight: 800, background: 'linear-gradient(to bottom, #ffffff, #a7f3d0)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', lineHeight: 1.1, marginBottom: 12 }}>
                        {Number(prediction).toFixed(2)} Lakhs
                      </h2>
                      <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px 24px', borderRadius: 30, display: 'inline-block' }}>
                        <p style={{ color: '#a7f3d0', fontSize: '1.1rem', fontWeight: 500, margin: 0 }}>
                           Exact Match: {new Intl.NumberFormat('en-LK', { style: 'currency', currency: 'LKR', maximumFractionDigits: 0 }).format(prediction * 100000)}
                        </p>
                      </div>
                   </div>
                </motion.div>
             )}

             {error && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1, marginTop: 40 }} exit={{ opacity: 0, scale: 0.95 }}
                >
                   <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 16, padding: 24, textAlign: 'center' }}>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: '#ef4444', fontWeight: 600, marginBottom: 8}}>
                        <AlertCircle size={18} /> Error Occurred
                      </div>
                      <p style={{ color: 'white' }}>{error}</p>
                   </div>
                </motion.div>
             )}
          </AnimatePresence>
        </motion.div>

        {/* AI Analytics Section */}
        <div id="analytics" style={{ width: '100%', maxWidth: 1200, padding: '0 24px', marginTop: 40 }}>
           <div style={{ textAlign: 'center', marginBottom: 48 }}>
             <h3 style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: 16 }}>Model Analytics & Accuracy</h3>
             <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', maxWidth: 600, margin: '0 auto' }}>Deep dive into the underlying metrics that power our high-precision automotive valuation engine.</p>
           </div>
           
           <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: 32 }}>
              
              <div className="glass-panel" style={{ padding: 32, display: 'flex', flexDirection: 'column', gap: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                   <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: 12, borderRadius: 12 }}><BarChart3 size={24} color="#3b82f6" /></div>
                   <h4 style={{ fontSize: '1.3rem', fontWeight: 700 }}>Feature Importance Matrix</h4>
                </div>
                <div style={{ background: 'white', borderRadius: 16, overflow: 'hidden', padding: 16, height: 360 }}>
                  <img src="/feature_importance.png" alt="Feature Importance" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Visualizing which car attributes carry the heaviest statistical weight in price determination.</p>
              </div>

              <div className="glass-panel" style={{ padding: 32, display: 'flex', flexDirection: 'column', gap: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                   <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: 12, borderRadius: 12 }}><TrendingUp size={24} color="#10b981" /></div>
                   <h4 style={{ fontSize: '1.3rem', fontWeight: 700 }}>Model Evaluation</h4>
                </div>
                <div style={{ background: 'white', borderRadius: 16, overflow: 'hidden', padding: 16, height: 360 }}>
                  <img src="/model_evaluation.png" alt="Model Evaluation" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>The predicted versus actual price map, demonstrating extremely low variance and high precision tracking.</p>
              </div>

              <div className="glass-panel" style={{ padding: 32, display: 'flex', flexDirection: 'column', gap: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                   <div style={{ background: 'rgba(245, 158, 11, 0.1)', padding: 12, borderRadius: 12 }}><ScatterChart size={24} color="#f59e0b" /></div>
                   <h4 style={{ fontSize: '1.3rem', fontWeight: 700 }}>Correlation Heatmap</h4>
                </div>
                <div style={{ background: 'white', borderRadius: 16, overflow: 'hidden', padding: 16, height: 360 }}>
                  <img src="/correlation_heatmap.png" alt="Correlation Heatmap" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>A deep variable-to-variable correlation mapping identifying collinearity dependencies.</p>
              </div>

              <div className="glass-panel" style={{ padding: 32, display: 'flex', flexDirection: 'column', gap: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                   <div style={{ background: 'rgba(236, 72, 153, 0.1)', padding: 12, borderRadius: 12 }}><PieChart size={24} color="#ec4899" /></div>
                   <h4 style={{ fontSize: '1.3rem', fontWeight: 700 }}>Price Distribution</h4>
                </div>
                <div style={{ background: 'white', borderRadius: 16, overflow: 'hidden', padding: 16, height: 360 }}>
                  <img src="/price_distribution.png" alt="Price Distribution" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>The spread and density curve of the entire historic car market base data.</p>
              </div>

           </div>
        </div>

      </main>
      
      {/* Footer */}
      <footer style={{ borderTop: '1px solid rgba(255,255,255,0.05)', background: 'rgba(8, 15, 30, 0.95)', padding: '40px 0', textAlign: 'center' }}>
         <div style={{ display: 'flex', justifyContent: 'center', gap: 8, color: 'var(--text-secondary)', alignItems: 'center' }}>
           <Car size={18} /> <span style={{ fontWeight: 600 }}>Autolytica ML Systems</span> &copy; 2026. Data precision guaranteed.
         </div>
      </footer>
      
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  );
}
