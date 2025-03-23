import { useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";
import { transcribeAudio } from "../services/api";

export default function Upload({ onTranscription }) {
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState("arabic");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an audio file first.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await transcribeAudio(file, language);
      onTranscription(result.transcription);
    } catch (err) {
      setError(err.message || "Failed to transcribe audio");
      console.error("Transcription error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      className="flex flex-col items-center space-y-6 p-8 bg-white/10 backdrop-blur-md rounded-2xl shadow-xl border border-white/20"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      {/* Language Selection */}
      <div className="w-full">
        <label className="text-black font-semibold mb-2 block">Select Language:</label>
        <select 
          value={language} 
          onChange={(e) => setLanguage(e.target.value)}
          className="w-full p-2 rounded-lg bg-white/20 text-black border border-white/30"
        >
          <option value="arabic">Arabic</option>
          <option value="hindi">Hindi</option>
        </select>
      </div>

      {/* Upload Area */}
      <label className="flex items-center justify-center w-72 h-40 border-2 border-dashed border-black rounded-xl cursor-pointer hover:bg-white/10 transition">
        <input type="file" accept="audio/*" onChange={handleFileChange} hidden />
        <div className="flex flex-col items-center">
          <UploadCloud size={48} className="text-black mb-2" />
          <p className="text-black font-semibold">Click to upload an audio file</p>
        </div>
      </label>

      {/* Display selected file name */}
      {file && (
        <p className="text-gray-300 text-sm italic">Selected: {file.name}</p>
      )}

      {/* Error message */}
      {error && (
        <p className="text-red-500 text-sm">{error}</p>
      )}

      {/* Upload Button */}
      <motion.button
        onClick={handleUpload}
        disabled={isLoading}
        className={`px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all ${
          isLoading ? "opacity-70 cursor-not-allowed" : ""
        }`}
        whileHover={{ scale: isLoading ? 1 : 1.1 }}
        whileTap={{ scale: isLoading ? 1 : 0.95 }}
      >
        {isLoading ? "Transcribing..." : "Upload & Transcribe"}
      </motion.button>
    </motion.div>
  );
}
