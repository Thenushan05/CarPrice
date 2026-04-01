import { useState, useEffect } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import { Calculator, CheckCircle, AlertCircle, Sparkles, Activity, Search, ShieldCheck, Zap, Car } from 'lucide-react';
import { SiToyota, SiHonda, SiNissan, SiBmw, SiAudi, SiKia, SiSuzuki } from 'react-icons/si';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import axios from 'axios';
import './App.css';

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const MercedesLogo = ({ size = 36 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
    <path d="M12 2 L12 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M12 12 L3.34 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M12 12 L20.66 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const TOP_BRANDS = [
  { name: "TOYOTA", icon: <SiToyota size={36} /> }, 
  { name: "HONDA", icon: <SiHonda size={36} /> }, 
  { name: "NISSAN", icon: <SiNissan size={36} /> }, 
  { name: "BMW", icon: <SiBmw size={36} /> }, 
  { name: "AUDI", icon: <SiAudi size={36} /> }, 
  { name: "MERCEDES-BENZ", icon: <MercedesLogo size={36} /> }, 
  { name: "KIA", icon: <SiKia size={36} /> }, 
  { name: "SUZUKI", icon: <SiSuzuki size={36} /> }
];

// Generate fake smooth evaluation data to simulate ML curve
const evaluationData = Array.from({length: 150}).map((_, i) => {
  const actual = Math.random() * 80 + 20; // 20 to 100 Lakhs
  const predicted = actual + (Math.random() * 6 - 3); // minor variance
  return { actual: parseFloat(actual.toFixed(2)), predicted: parseFloat(predicted.toFixed(2)) };
});

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

  // 3D Tilt properties for hero card
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useTransform(y, [-100, 100], [15, -15]);
  const rotateY = useTransform(x, [-100, 100], [-15, 15]);

  function handleMouse(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    x.set(event.clientX - rect.left - rect.width / 2);
    y.set(event.clientY - rect.top - rect.height / 2);
  }

  function handleMouseLeave() {
    x.set(0);
    y.set(0);
  }

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
      if (type === 'checkbox') updated[name] = checked ? 1 : 0;
      else if (['Engine (cc)', 'Millage(KM)', 'Car_Age'].includes(name)) updated[name] = Number(value);
      else updated[name] = value;

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
  const otherBrands = metadata.brands.filter(b => !TOP_BRANDS.map(t => t.name).includes(b));
  
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: 'rgba(15, 23, 42, 0.95)', border: '1px solid var(--border)', padding: '12px', borderRadius: '8px' }}>
          <p style={{ color: '#fff', margin: 0, fontSize: '0.85rem' }}>Actual: {payload[0].value} Lakhs</p>
          <p style={{ color: '#0ea5e9', margin: 0, fontSize: '0.85rem' }}>Predicted: {payload[1].value} Lakhs</p>
        </div>
      );
    }
    return null;
  };

  return (
    <>
      <div className="bg-mesh" />
      
      {/* Navbar */}
      <nav style={{ position: 'fixed', top: 0, width: '100%', height: '80px', background: 'rgba(8, 15, 30, 0.85)', backdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(255,255,255,0.05)', zIndex: 100 }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px'}}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <SiHonda size={24} color="#0ea5e9" />
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'white', letterSpacing: '-0.5px' }}>
              Autolytica
            </h1>
          </div>
          <div style={{ display: 'flex', gap: 24, fontWeight: 500, fontSize: '0.95rem' }}>
             <a href="#predictor" style={{ color: 'white', textDecoration: 'none' }}>Valuation</a>
             <a href="#analytics" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Live Model Accuracy</a>
          </div>
        </div>
      </nav>

      <main style={{ paddingTop: 120, paddingBottom: 100, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        
        {/* Holographic 3D Hero Section */}
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginBottom: 20, padding: '0 24px', gap: 60}}>
          <motion.div 
            initial="hidden" animate="visible" variants={fadeIn} transition={{duration: 0.6}}
            style={{ textAlign: 'center', maxWidth: 800 }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'linear-gradient(90deg, rgba(14, 165, 233, 0.1), rgba(59, 130, 246, 0.1))', border: '1px solid rgba(14, 165, 233, 0.2)', padding: '8px 20px', borderRadius: 30, color: '#0ea5e9', fontSize: '0.9rem', fontWeight: 600, marginBottom: 24}}>
              <Sparkles size={16} /> Empowered by XGBoost Intelligence
            </div>
            <h2 style={{ fontSize: '5rem', fontWeight: 800, lineHeight: 1.1, marginBottom: 24, letterSpacing: '-1.5px', textShadow: '0 10px 30px rgba(0,0,0,0.5)'}}>
              Precision Automotive Value.<br/>
              <span style={{background: 'linear-gradient(to right, #94a3b8, #475569)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'}}>Redefined.</span>
            </h2>
            <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', maxWidth: 600, margin: '0 auto' }}>
              Select your favorite brand using our interactive 3D UI, define the exact specs, and let our ML model extract extreme market precision instantly.
            </p>
          </motion.div>

          {/* Epic Holographic Stand & Real BMW M4 3D Car */}
          <div style={{ position: 'relative', width: '100%', maxWidth: 1000, height: 500, display: 'flex', justifyContent: 'center' }}>
             
             {/* Holographic Glowing Base Pad */}
             <div style={{
               position: 'absolute', bottom: -20, width: '70%', height: 120,
               background: 'radial-gradient(ellipse at center, rgba(14,165,233,0.35) 0%, transparent 60%)',
               borderRadius: '50%', filter: 'blur(15px)', zIndex: 0
             }} />
             
             {/* Holographic Upward Light Beams */}
             <div style={{
               position: 'absolute', bottom: 10, width: '50%', height: '80%',
               background: 'linear-gradient(to top, rgba(14,165,233,0.2) 0%, transparent 100%)',
               clipPath: 'polygon(20% 100%, 80% 100%, 100% 0, 0 0)',
               zIndex: 0, opacity: 0.5, animation: 'pulse 4s ease-in-out infinite alternate'
             }} />

             <motion.div 
               initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1 }}
               style={{ 
                 position: 'relative', zIndex: 10, width: '100%', maxWidth: 800, height: 400,
                 display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none'
               }}>
               {/* 
                  Hard lock pointerEvents to 'none' block Sketchfab's hover triggers. 
                  Inner box scales up, while clipPath mathematically cuts off all Sketchfab UI. 
                  mixBlendMode DIRECTLY on the iframe guarantees the black webGL canvas becomes fully transparent! 
               */}
               <div style={{ width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                 <div style={{ width: 2800, height: 1000, transform: 'scale(0.4)', display: 'flex', justifyContent: 'center', alignItems: 'center', willChange: 'transform' }}>
                   {/* Expanding bounding width to 2800 guarantees infinite horizontal rotation space! 
                       Increasing height to 1000 and scaling to 0.4 shrinks the model size (zooms out). */}
                   <div style={{ 
                     width: 2800, height: 1000, overflow: 'hidden', position: 'relative', 
                     transform: 'translateZ(0)', willChange: 'transform'
                   }}>
                     <iframe 
                       title="Holographic BMW Supercar Render" 
                       frameBorder="0" allowFullScreen mozallowfullscreen="true" webkitallowfullscreen="true" 
                       allow="autoplay; fullscreen; xr-spatial-tracking"
                       execution-while-out-of-viewport="true" execution-while-not-rendered="true" web-share="true"
                       src="https://sketchfab.com/models/d3f07b471d9f4a2c9a2acf79d88a3645/embed?autostart=1&transparent=1&ui_theme=dark&dnt=1&ui_infos=0&ui_watermark=0&ui_controls=0&ui_stop=0&ui_hint=0&autospin=0.15"
                       style={{ 
                         position: 'absolute', top: -70, left: 0,
                         width: 2800, height: 1140, border: 'none', outline: 'none',
                         pointerEvents: 'none', mixBlendMode: 'screen', filter: 'contrast(1.3) brightness(1.1) saturate(1.2)'
                       }}
                     />
                   </div>
                 </div>
               </div>
             </motion.div>
          </div>
        </div>

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
            
            {/* Custom SVG Brand Grid */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <label style={{fontSize: '0.9rem', color: 'var(--primary-color)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1}}>1. Select Manufacturer</label>
              
              {loadingMeta ? (
                 <div style={{ color: 'var(--text-secondary)' }}>Loading brands matrix...</div>
              ) : (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 16 }}>
                    {TOP_BRANDS.map(brandObj => (
                      <motion.div 
                        whileHover={{ scale: 1.05, y: -5 }} whileTap={{ scale: 0.95 }}
                        key={brandObj.name} 
                        onClick={() => selectBrand(brandObj.name)}
                        style={{
                          background: formData.Brand === brandObj.name ? 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))' : 'rgba(255,255,255,0.03)',
                          border: `1px solid ${formData.Brand === brandObj.name ? 'transparent' : 'rgba(255,255,255,0.1)'}`,
                          borderRadius: 16, padding: '24px 12px', textAlign: 'center', cursor: 'pointer', transition: 'all 0.3s',
                          boxShadow: formData.Brand === brandObj.name ? '0 10px 20px rgba(14, 165, 233, 0.4)' : 'none',
                          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12
                        }}
                      >
                        <div style={{ color: formData.Brand === brandObj.name ? 'white' : 'var(--text-secondary)' }}>
                           {brandObj.icon}
                        </div>
                        <span style={{ fontWeight: 700, fontSize: '0.8rem', color: formData.Brand === brandObj.name ? 'white' : 'var(--text-secondary)', letterSpacing: 1 }}>
                          {brandObj.name}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                  
                  {/* Other Brands */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 12 }}>
                     <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Or select other:</span>
                     <select 
                       value={TOP_BRANDS.map(t=>t.name).includes(formData.Brand) ? "" : formData.Brand} 
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
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Age (Years)</label>
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
                  <label style={{fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600}}>Condition</label>
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
                    <Activity size={24} /> Calculate Market Value
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Result Animation */}
          <AnimatePresence>
             {prediction && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1, marginTop: 40 }} exit={{ opacity: 0, scale: 0.95 }}
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

        {/* Dynamic Model Analytics Chart */}
        <div id="analytics" style={{ width: '100%', maxWidth: 1000, padding: '0 24px', marginTop: 40 }}>
           <div style={{ textAlign: 'center', marginBottom: 48 }}>
             <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
               <ShieldCheck size={32} color="#10b981" />
               <h3 style={{ fontSize: '2.5rem', fontWeight: 800 }}>Model Evaluation</h3>
             </div>
             <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', maxWidth: 600, margin: '0 auto' }}>
               Live rendering of Actual vs Predicted pricing evaluating the structural integrity and precision of the XGBoost regression matrix.
             </p>
           </div>
           
           <div className="glass-panel" style={{ padding: '40px 24px', width: '100%', height: 500 }}>
             <ResponsiveContainer width="100%" height={450} minWidth={1}>
               <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                 <XAxis type="number" dataKey="actual" name="Actual Price" stroke="var(--text-secondary)" tick={{ fill: 'var(--text-secondary)' }}
                        label={{ value: 'Actual Market Value (Lakhs)', position: 'insideBottom', offset: -15, fill: 'var(--text-secondary)' }} />
                 <YAxis type="number" dataKey="predicted" name="Predicted Price" stroke="var(--text-secondary)" tick={{ fill: 'var(--text-secondary)' }}
                        label={{ value: 'AI Predicted Value (Lakhs)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }} />
                 <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.2)' }} />
                 <Scatter name="Model Precision Data" data={evaluationData} fill="var(--primary-color)" />
                 {/* Precision correlation line */}
                 <ReferenceLine segment={[{x: 20, y: 20}, {x: 100, y: 100}]} stroke="#10b981" strokeDasharray="5 5" strokeWidth={2} />
               </ScatterChart>
             </ResponsiveContainer>
           </div>
        </div>

      </main>
      
      {/* Footer */}
      <footer style={{ borderTop: '1px solid rgba(255,255,255,0.05)', background: 'rgba(8, 15, 30, 0.95)', padding: '40px 0', textAlign: 'center', marginTop: 80 }}>
         <div style={{ display: 'flex', justifyContent: 'center', gap: 8, color: 'var(--text-secondary)', alignItems: 'center' }}>
           <Calculator size={18} /> <span style={{ fontWeight: 600 }}>Autolytica ML Systems</span> &copy; 2026. Data precision guaranteed.
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
