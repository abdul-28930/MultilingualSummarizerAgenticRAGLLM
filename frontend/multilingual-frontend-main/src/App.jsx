import { useState } from "react";
import { motion } from "framer-motion";
import { Parallax } from "react-scroll-parallax";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import Upload from "./components/Upload";
import Summarization from "./pages/Summarization";
import Query from "./pages/Query";
import ApiStatus from "./components/ApiStatus";
import { Tranquiluxe } from "uvcanvas"; // Background animation
import "./styles/global.css"; // Ensure global styles are included



function Home() {
  const [transcription, setTranscription] = useState("");
  const navigate = useNavigate();

  return (
    <div className="relative w-full min-h-screen flex flex-col items-center justify-center text-center font-poppins">
      {/* Background Animation */}
      <div className="absolute inset-0 -z-10">
        <Tranquiluxe speed={1.2} />
      </div>

      {/* Title - "ZETA" with Gold & White Gradient */}
      <Parallax speed={-10}>
        <motion.h1
          className="text-7xl md:text-9xl font-extrabold mb-6"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          whileHover={{ scale: 1.05 }}
          style={{
            fontFamily: "Poppins, sans-serif",
            background: "linear-gradient(180deg, #ffffff, #ffd700)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            textShadow: `0 0 10px rgba(255, 255, 255, 0.8), 0 0 20px rgba(255, 215, 0, 0.8)`,
          }}
        >
          ZETA
        </motion.h1>
      </Parallax>

      {/* Black & White Title */}
      <motion.h2
        className="text-3xl md:text-5xl font-semibold text-gray-100 drop-shadow-md mt-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        Multilingual Summarization Tool 🌍📜
      </motion.h2>

      {/* Spacing between Title & Description */}
      <div className="mt-6"></div>

      {/* Enhanced Description with Glow & Animation */}
      <motion.p
        className="text-black max-w-lg text-lg font-bold drop-shadow-lg px-4"
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3 }}
        style={{
          textShadow: "2px 2px 5px rgba(0,0,0,0.5), 0 0 10px rgba(0,0,0,0.3)", // Added glow
        }}
      >
        Convert speech to text, translate, summarize, and query your summaries
        in multiple languages—all in one powerful tool.
      </motion.p>

      {/* Upload Component */}
      <Upload onTranscription={setTranscription} />

      {/* Transcribed Text Display */}
      {transcription && (
        <motion.div
          className="mt-6 p-6 bg-white/10 backdrop-blur-md rounded-xl shadow-xl max-w-2xl border border-white/20"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-xl font-bold text-yellow-400 drop-shadow-lg">Transcribed Text:</h2>
          <p className="text-gray-200 mt-2">{transcription}</p>
        </motion.div>
      )}

      {/* Summarization Button */}
      {transcription && (
        <motion.button
          onClick={() => navigate("/summarization", { state: { transcription } })}
          className="mt-6 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          Summarization
        </motion.button>
      )}
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/summarization" element={<Summarization />} />
        <Route path="/query" element={<Query />} />
      </Routes>
      <ApiStatus />
    </Router>
  );
}
