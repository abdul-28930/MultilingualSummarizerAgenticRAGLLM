import { useState } from "react";
import { motion } from "framer-motion";
import { generateFlowchart } from "../services/api";

export default function FlowchartViewer({ text }) {
  const [flowchartData, setFlowchartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("keyword"); // "keyword" or "logical"

  const handleGenerateFlowchart = async () => {
    if (!text) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await generateFlowchart(text);
      setFlowchartData(data);
    } catch (err) {
      setError(err.message || "Failed to generate flowchart");
      console.error("Flowchart error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white/10 backdrop-blur-md rounded-xl shadow-2xl p-6 space-y-4 border border-white/20">
      <h2 className="text-2xl font-bold text-purple-400 drop-shadow-lg">Keyword Flowchart</h2>
      
      {!flowchartData && !loading && (
        <motion.button
          onClick={handleGenerateFlowchart}
          className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Generate Flowchart
        </motion.button>
      )}
      
      {loading && (
        <div className="flex justify-center items-center py-10">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-purple-500"></div>
        </div>
      )}
      
      {error && (
        <div className="bg-red-500/20 border border-red-500 text-white p-4 rounded-lg">
          {error}
        </div>
      )}
      
      {flowchartData && (
        <div className="space-y-4">
          <div className="flex space-x-2 border-b border-gray-700 pb-2">
            <button
              className={`px-4 py-2 rounded-t-lg ${
                activeTab === "keyword" 
                  ? "bg-purple-500 text-white" 
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
              onClick={() => setActiveTab("keyword")}
            >
              Keyword Flowchart
            </button>
            <button
              className={`px-4 py-2 rounded-t-lg ${
                activeTab === "logical" 
                  ? "bg-purple-500 text-white" 
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
              onClick={() => setActiveTab("logical")}
            >
              Logical Flow
            </button>
          </div>
          
          <div className="bg-white rounded-lg p-2 overflow-auto max-h-[500px]">
            {activeTab === "keyword" ? (
              <img 
                src={`data:image/png;base64,${flowchartData.keyword_flowchart}`} 
                alt="Keyword Flowchart" 
                className="mx-auto"
              />
            ) : (
              <img 
                src={`data:image/png;base64,${flowchartData.logical_flowchart}`} 
                alt="Logical Flowchart" 
                className="mx-auto"
              />
            )}
          </div>
          
          <div className="mt-4">
            <h3 className="text-xl font-semibold text-purple-300 mb-2">Extracted Keywords</h3>
            <div className="flex flex-wrap gap-2">
              {flowchartData.keywords.map((item, index) => (
                <div 
                  key={index} 
                  className="bg-gray-700 text-white px-3 py-1 rounded-full text-sm"
                >
                  {item.keyword} ({item.score.toFixed(4)})
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
