import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate, useLocation } from "react-router-dom";
import { generateSummary, translateText } from "../services/api";
import FlowchartViewer from "../components/FlowchartViewer";

export default function Summarization() {
  const location = useLocation();
  const transcription = location.state?.transcription || "No transcription available.";
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [translationLoading, setTranslationLoading] = useState(false);
  const [translatedSummary, setTranslatedSummary] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("English");
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [showFlowchart, setShowFlowchart] = useState(false);
  const navigate = useNavigate();

  const handleSummarization = async (type) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await generateSummary(transcription, type.toLowerCase());
      setSummary(result.summary);
      setMetrics(result.metrics);
    } catch (err) {
      setError(err.message || `Failed to generate ${type.toLowerCase()} summary`);
      console.error("Summarization error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleTranslation = async () => {
    if (!summary) return;
    
    setTranslationLoading(true);
    setError(null);
    
    try {
      const result = await translateText(summary, targetLanguage);
      setTranslatedSummary(result.translated_text);
    } catch (err) {
      setError(err.message || "Failed to translate summary");
      console.error("Translation error:", err);
    } finally {
      setTranslationLoading(false);
    }
  };

  const toggleFlowchart = () => {
    setShowFlowchart(!showFlowchart);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-950 text-white text-center space-y-6 font-poppins">
      <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
        Summarization
      </h1>

      <div className="p-6 bg-white/10 backdrop-blur-md rounded-xl shadow-xl max-w-2xl border border-white/20">
        <h2 className="text-lg font-bold text-blue-400">Transcribed Text:</h2>
        <p className="text-gray-300 mt-2">{transcription}</p>
      </div>

      {error && (
        <div className="p-4 bg-red-500/20 border border-red-500 rounded-xl max-w-2xl">
          <p className="text-red-300">{error}</p>
        </div>
      )}

      <div className="flex space-x-4">
        <motion.button
          onClick={() => handleSummarization("Extractive")}
          disabled={loading}
          className={`px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all ${
            loading ? "opacity-70 cursor-not-allowed" : ""
          }`}
          whileHover={{ scale: loading ? 1 : 1.1 }}
          whileTap={{ scale: loading ? 1 : 0.95 }}
        >
          Extractive Summarization
        </motion.button>

        <motion.button
          onClick={() => handleSummarization("Abstractive")}
          disabled={loading}
          className={`px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all ${
            loading ? "opacity-70 cursor-not-allowed" : ""
          }`}
          whileHover={{ scale: loading ? 1 : 1.1 }}
          whileTap={{ scale: loading ? 1 : 0.95 }}
        >
          Abstractive Summarization
        </motion.button>
      </div>

      {loading && <p className="text-gray-400">Generating summary...</p>}

      {summary && (
        <motion.div
          className="mt-6 p-6 bg-white/10 backdrop-blur-md rounded-xl shadow-xl max-w-2xl border border-white/20"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-lg font-bold text-blue-400">Summary:</h2>
          <p className="text-gray-300 mt-2">{summary}</p>

          {/* Translation Controls */}
          <div className="mt-4 flex items-center gap-2">
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              className="p-2 rounded-lg bg-gray-800 text-white border border-gray-600"
            >
              <option value="English">English</option>
              <option value="Arabic">Arabic</option>
              <option value="Hindi">Hindi</option>
              <option value="Spanish">Spanish</option>
              <option value="French">French</option>
              <option value="German">German</option>
              <option value="Chinese">Chinese</option>
              <option value="Japanese">Japanese</option>
            </select>
            
            <motion.button
              onClick={handleTranslation}
              disabled={translationLoading}
              className={`px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-semibold rounded-lg shadow-lg hover:scale-105 hover:shadow-xl transition-all ${
                translationLoading ? "opacity-70 cursor-not-allowed" : ""
              }`}
              whileHover={{ scale: translationLoading ? 1 : 1.05 }}
              whileTap={{ scale: translationLoading ? 1 : 0.95 }}
            >
              {translationLoading ? "Translating..." : "Translate"}
            </motion.button>
          </div>

          {/* Translated Summary */}
          {translatedSummary && (
            <div className="mt-4 p-4 bg-indigo-500/20 border border-indigo-500/50 rounded-lg">
              <h3 className="text-md font-bold text-indigo-300">Translated Summary ({targetLanguage}):</h3>
              <p className="text-gray-300 mt-2">{translatedSummary}</p>
            </div>
          )}

          {/* Metrics Display */}
          {metrics && (
            <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <h3 className="text-md font-bold text-blue-300">Summary Metrics:</h3>
              <div className="grid grid-cols-2 gap-2 mt-2">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key} className="text-left">
                    <span className="text-gray-400">{key}: </span>
                    <span className="text-gray-300">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-center items-center gap-4 mt-4">
            <motion.button
              onClick={() => navigate("/query", { state: { summary } })}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              Query the Summary
            </motion.button>

            <motion.button
              onClick={toggleFlowchart}
              className="px-6 py-3 bg-gradient-to-r from-green-500 to-teal-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              Flowchart
            </motion.button>
          </div>
        </motion.div>
      )}
      
      {/* Flowchart Viewer */}
      {showFlowchart && summary && (
        <motion.div
          className="w-full max-w-4xl mt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <FlowchartViewer text={summary} />
        </motion.div>
      )}
    </div>
  );
}
