import React, { useState } from 'react';
import { Code2, Network, Database, Play, Download, CheckCircle2, FileJson, FileSpreadsheet } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import * as XLSX from 'xlsx';

const PYTHON_CODE = `import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================================
# 1) CARICAMENTO CSV
# ============================================================

df = pd.read_csv("c:/Users/Utente/Desktop/enhanced_customer_data.csv")

# ============================================================
# 2) DEFINIZIONE COLONNE
# ============================================================

numeric_features = [
    "Age", "Purchase Amount (USD)", "Previous Purchases",
    "Frequency of Purchases", "Loyalty Score", "Engagement Score",
    "Spending Propensity", "Rating Stability", "Purchase Intensity"
]

categorical_features = [
    "Gender", "Item Purchased", "Category", "Location", "Size",
    "Color", "Season", "Payment Method", "Preferred Payment Method",
    "Category Group"
]

binary_features = [
    "Subscription Status", "Discount Applied", "Promo Code Used",
    "Digital Payment", "Recurring Buyer"
]

# ============================================================
# 3) CONVERSIONE BINARIE
# ============================================================

for col in binary_features:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].map({
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "active": 1, "inactive": 0,
        "1": 1, "0": 0
    }).fillna(0).astype(int)

# ============================================================
# 4) PREPROCESSING
# ============================================================

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)

finale = preprocessor.fit_transform(df)
X = finale.toarray() if hasattr(finale, "toarray") else finale

input_dim = X.shape[1]
latent_dim = 12

print("Shape finale:", X.shape)

# ============================================================
# 5) ENCODER
# ============================================================

inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(128, activation="relu")(inputs)
h = layers.Dense(64, activation="relu")(h)
h = layers.Dense(32, activation="relu")(h)

z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

# ============================================================
# 6) DECODER
# ============================================================

decoder_input = layers.Input(shape=(latent_dim,))
d = layers.Dense(32, activation="relu")(decoder_input)
d = layers.Dense(64, activation="relu")(d)
d = layers.Dense(128, activation="relu")(d)
outputs = layers.Dense(input_dim, activation="sigmoid")(d)

decoder = Model(decoder_input, outputs, name="decoder")

# ============================================================
# 7) VAE CUSTOM (COMPATIBILE TF 2.15)
# ============================================================

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

# ============================================================
# 8) ISTANZA E COMPILAZIONE
# ============================================================

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")

# ============================================================
# 9) TRAINING
# ============================================================

vae.fit(X, epochs=80, batch_size=32, validation_split=0.2)

# ============================================================
# 10) SALVATAGGIO
# ============================================================

encoder.save("encoder_model.keras")
decoder.save("decoder_model.keras")
vae.save("vae_model.keras")
joblib.dump(preprocessor, "preprocessor.pkl")

# ============================================================
# 11) GENERAZIONE SINTETICA
# ============================================================

def generate_customers(n=10):
    z_new = np.random.normal(size=(n, latent_dim))
    return decoder.predict(z_new)

# ============================================================
# 12) DECODIFICA
# ============================================================

def decode_synthetic(synth_matrix, preprocessor, original_columns):
    decoded = preprocessor.inverse_transform(synth_matrix)
    df_decoded = pd.DataFrame(decoded, columns=original_columns)
    return df_decoded

# ============================================================
# 13) TEST
# ============================================================

synthetic = generate_customers(5)
decoded = decode_synthetic(synthetic, preprocessor, df.columns)

print(decoded.head())`;

type Tab = 'code' | 'architecture' | 'simulation';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('code');
  const [syntheticData, setSyntheticData] = useState<any[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [numToGenerate, setNumToGenerate] = useState<number>(10);

  const exportToCSV = () => {
    if (syntheticData.length === 0) return;
    
    // Get headers (excluding the internal 'id')
    const headers = Object.keys(syntheticData[0]).filter(k => k !== 'id');
    
    // Create CSV content
    const csvContent = [
      headers.join(','),
      ...syntheticData.map(row => headers.map(h => `"${row[h]}"`).join(','))
    ].join('\n');
    
    // Trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `clienti_sintetici_${syntheticData.length}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToExcel = () => {
    if (syntheticData.length === 0) return;
    
    // Rimuovi l'ID interno dai dati prima di esportare
    const dataToExport = syntheticData.map(({ id, ...rest }) => rest);
    
    // Crea il foglio di lavoro e la cartella di lavoro
    const worksheet = XLSX.utils.json_to_sheet(dataToExport);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Clienti Sintetici");
    
    // Salva il file
    XLSX.writeFile(workbook, `clienti_sintetici_${syntheticData.length}.xlsx`);
  };

  const generateMockData = (n: number) => {
    setIsGenerating(true);
    setTimeout(() => {
      const genders = ["Uomo", "Donna", "Non binario"];
      const categories = ["Elettronica", "Abbigliamento", "Casa", "Bellezza", "Sport"];
      const locations = ["Roma", "Milano", "Napoli", "Torino", "Firenze"];
      const payments = ["Carta di Credito", "PayPal", "Apple Pay", "Criptovalute"];
      
      const newData = Array.from({ length: n }).map((_, i) => ({
        id: Math.random().toString(36).substr(2, 9),
        Age: Math.floor(Math.random() * 50) + 18,
        "Purchase Amount (USD)": (Math.random() * 800 + 20).toFixed(2),
        "Previous Purchases": Math.floor(Math.random() * 50),
        "Loyalty Score": (Math.random() * 10).toFixed(1),
        Gender: genders[Math.floor(Math.random() * genders.length)],
        Category: categories[Math.floor(Math.random() * categories.length)],
        Location: locations[Math.floor(Math.random() * locations.length)],
        "Payment Method": payments[Math.floor(Math.random() * payments.length)],
        "Subscription Status": Math.random() > 0.6 ? "Sì" : "No",
        "Discount Applied": Math.random() > 0.5 ? "Sì" : "No",
      }));
      
      setSyntheticData(newData);
      setIsGenerating(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white">
              <Network size={20} />
            </div>
            <h1 className="text-xl font-semibold tracking-tight text-slate-900">Generatore Clienti VAE</h1>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="flex items-center gap-1"><CheckCircle2 size={16} className="text-emerald-500" /> TF 2.15</span>
            <span className="flex items-center gap-1 ml-4"><CheckCircle2 size={16} className="text-emerald-500" /> Scikit-Learn</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tabs */}
        <div className="flex space-x-1 bg-slate-200/50 p-1 rounded-xl w-fit mb-8">
          <button
            onClick={() => setActiveTab('code')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'code' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
          >
            <Code2 size={18} />
            Sorgente Python
          </button>
          <button
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'architecture' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
          >
            <Network size={18} />
            Architettura
          </button>
          <button
            onClick={() => setActiveTab('simulation')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'simulation' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
          >
            <Database size={18} />
            Simulazione
          </button>
        </div>

        {/* Content */}
        <AnimatePresence mode="wait">
          {activeTab === 'code' && (
            <motion.div
              key="code"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-slate-900 rounded-2xl shadow-xl overflow-hidden border border-slate-800"
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/50">
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-rose-500"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                </div>
                <span className="text-xs font-mono text-slate-400">vae_model.py</span>
              </div>
              <div className="p-6 overflow-x-auto">
                <pre className="text-sm font-mono text-slate-300 leading-relaxed">
                  <code>{PYTHON_CODE}</code>
                </pre>
              </div>
            </motion.div>
          )}

          {activeTab === 'architecture' && (
            <motion.div
              key="architecture"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid grid-cols-1 md:grid-cols-3 gap-6"
            >
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 col-span-1 md:col-span-3">
                <h2 className="text-xl font-semibold mb-4">Panoramica della Pipeline</h2>
                <p className="text-slate-600 mb-6">
                  Questo codice implementa un Autoencoder Varianzionale (VAE) per apprendere la distribuzione sottostante di un dataset di clienti e generare profili cliente sintetici realistici.
                </p>
                
                <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-center">
                  <div className="flex-1 bg-slate-50 p-4 rounded-xl border border-slate-100 w-full">
                    <Database className="mx-auto mb-2 text-indigo-500" />
                    <h3 className="font-medium text-slate-900">1. Pre-elaborazione</h3>
                    <p className="text-xs text-slate-500 mt-1">MinMaxScaler per i numerici, OneHotEncoder per i categorici.</p>
                  </div>
                  <div className="hidden md:block text-slate-300">→</div>
                  <div className="flex-1 bg-slate-50 p-4 rounded-xl border border-slate-100 w-full">
                    <Network className="mx-auto mb-2 text-indigo-500" />
                    <h3 className="font-medium text-slate-900">2. Encoder</h3>
                    <p className="text-xs text-slate-500 mt-1">Comprime l'input in uno spazio latente a 12 dimensioni (media e varianza).</p>
                  </div>
                  <div className="hidden md:block text-slate-300">→</div>
                  <div className="flex-1 bg-slate-50 p-4 rounded-xl border border-slate-100 w-full">
                    <FileJson className="mx-auto mb-2 text-indigo-500" />
                    <h3 className="font-medium text-slate-900">3. Decoder</h3>
                    <p className="text-xs text-slate-500 mt-1">Ricostruisce i dati dallo spazio latente usando l'attivazione sigmoide.</p>
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <h3 className="font-semibold text-slate-900 mb-2">Feature Elaborate</h3>
                <ul className="space-y-2 text-sm text-slate-600">
                  <li><span className="font-medium text-slate-900">9 Numeriche:</span> Età, Importo Acquisto, Punteggio Fedeltà...</li>
                  <li><span className="font-medium text-slate-900">10 Categoriche:</span> Genere, Categoria, Posizione...</li>
                  <li><span className="font-medium text-slate-900">5 Binarie:</span> Abbonamento, Codice Promo...</li>
                </ul>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <h3 className="font-semibold text-slate-900 mb-2">Architettura di Rete</h3>
                <ul className="space-y-2 text-sm text-slate-600">
                  <li><span className="font-medium text-slate-900">Encoder:</span> 128 → 64 → 32 → 12 (Latente)</li>
                  <li><span className="font-medium text-slate-900">Decoder:</span> 12 (Latente) → 32 → 64 → 128 → Output</li>
                  <li><span className="font-medium text-slate-900">Loss:</span> Ricostruzione + Divergenza KL</li>
                </ul>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <h3 className="font-semibold text-slate-900 mb-2">Dettagli Addestramento</h3>
                <ul className="space-y-2 text-sm text-slate-600">
                  <li><span className="font-medium text-slate-900">Ottimizzatore:</span> Adam</li>
                  <li><span className="font-medium text-slate-900">Epoche:</span> 80</li>
                  <li><span className="font-medium text-slate-900">Dimensione Batch:</span> 32</li>
                  <li><span className="font-medium text-slate-900">Split Validazione:</span> 20%</li>
                </ul>
              </div>
            </motion.div>
          )}

          {activeTab === 'simulation' && (
            <motion.div
              key="simulation"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6"
            >
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Generatore Dati Sintetici</h2>
                  <p className="text-sm text-slate-500">Simula il decoder VAE generando nuovi profili cliente da vettori latenti casuali.</p>
                </div>
                <div className="flex items-center gap-3">
                  <select
                    value={numToGenerate}
                    onChange={(e) => setNumToGenerate(Number(e.target.value))}
                    disabled={isGenerating}
                    className="bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-xl focus:ring-indigo-500 focus:border-indigo-500 block px-4 py-3 outline-none cursor-pointer"
                  >
                    <option value={5}>5 Clienti</option>
                    <option value={10}>10 Clienti</option>
                    <option value={50}>50 Clienti</option>
                    <option value={100}>100 Clienti</option>
                    <option value={500}>500 Clienti</option>
                    <option value={1000}>1000 Clienti</option>
                  </select>
                  <button
                    onClick={() => generateMockData(numToGenerate)}
                    disabled={isGenerating}
                    className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-all disabled:opacity-70 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    {isGenerating ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <Play size={18} />
                    )}
                    {isGenerating ? 'Generazione...' : 'Genera'}
                  </button>
                </div>
              </div>

              {syntheticData.length > 0 && (
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-slate-500 uppercase bg-slate-50 border-b border-slate-200">
                        <tr>
                          <th className="px-6 py-4 font-medium">Età</th>
                          <th className="px-6 py-4 font-medium">Genere</th>
                          <th className="px-6 py-4 font-medium">Posizione</th>
                          <th className="px-6 py-4 font-medium">Categoria</th>
                          <th className="px-6 py-4 font-medium">Importo ($)</th>
                          <th className="px-6 py-4 font-medium">Fedeltà</th>
                          <th className="px-6 py-4 font-medium">Abbonato</th>
                        </tr>
                      </thead>
                      <tbody>
                        {syntheticData.map((row, idx) => (
                          <motion.tr 
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            key={row.id} 
                            className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors"
                          >
                            <td className="px-6 py-4 font-mono text-slate-600">{row.Age}</td>
                            <td className="px-6 py-4 text-slate-700">{row.Gender}</td>
                            <td className="px-6 py-4 text-slate-700">{row.Location}</td>
                            <td className="px-6 py-4 text-slate-700">
                              <span className="inline-flex items-center px-2 py-1 rounded-md bg-slate-100 text-xs font-medium text-slate-600">
                                {row.Category}
                              </span>
                            </td>
                            <td className="px-6 py-4 font-mono text-slate-600">${row["Purchase Amount (USD)"]}</td>
                            <td className="px-6 py-4 font-mono text-slate-600">{row["Loyalty Score"]}</td>
                            <td className="px-6 py-4">
                              <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${row["Subscription Status"] === 'Sì' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                                {row["Subscription Status"]}
                              </span>
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="p-4 border-t border-slate-200 bg-slate-50 flex justify-between items-center">
                    <span className="text-sm text-slate-500">Mostrando {syntheticData.length} clienti generati</span>
                    <div className="flex gap-2">
                      <button 
                        onClick={exportToCSV}
                        className="flex items-center gap-2 text-sm font-medium text-indigo-600 hover:text-indigo-700 transition-colors bg-indigo-50 px-4 py-2 rounded-lg border border-indigo-100 hover:bg-indigo-100"
                      >
                        <Download size={16} />
                        CSV
                      </button>
                      <button 
                        onClick={exportToExcel}
                        className="flex items-center gap-2 text-sm font-medium text-emerald-600 hover:text-emerald-700 transition-colors bg-emerald-50 px-4 py-2 rounded-lg border border-emerald-100 hover:bg-emerald-100"
                      >
                        <FileSpreadsheet size={16} />
                        Excel
                      </button>
                    </div>
                  </div>
                </div>
              )}
              
              {syntheticData.length === 0 && !isGenerating && (
                <div className="bg-slate-50 border border-slate-200 border-dashed rounded-2xl p-12 text-center">
                  <Database className="mx-auto h-12 w-12 text-slate-300 mb-4" />
                  <h3 className="text-sm font-medium text-slate-900">Nessun dato generato ancora</h3>
                  <p className="text-sm text-slate-500 mt-1">Clicca il pulsante genera per simulare il decoder VAE.</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
