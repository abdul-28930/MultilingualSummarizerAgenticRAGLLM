import { useLocation } from "react-router-dom";
import { useState } from "react";
import { motion } from "framer-motion";
import { querySummary } from "../services/api";

export default function Query() {
  const location = useLocation();
  const summary = location.state?.summary || "No summary available.";

  const [messages, setMessages] = useState([
    { text: `Summary: ${summary}`, sender: "system" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);
    setInput(""); // Clear input field
    setLoading(true);
    setError(null);

    try {
      // Call the API to get a response
      const result = await querySummary(input, summary);
      
      // Add system response to chat
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: result.response, sender: "system" },
      ]);
    } catch (err) {
      setError(err.message || "Failed to get a response");
      console.error("Query error:", err);
      
      // Add error message to chat
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: `Error: ${err.message || "Failed to get a response"}`, sender: "system" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#0f0f1a] text-white text-center space-y-6 font-['Poppins']">
      <h1 className="text-4xl font-bold text-purple-400 drop-shadow-lg">Query the Summary</h1>
      
      {/* Chat Box */}
      <div className="w-full max-w-2xl bg-white/10 backdrop-blur-md rounded-xl shadow-2xl p-6 space-y-4 border border-white/20 overflow-y-auto max-h-[400px]">
        {messages.map((msg, index) => (
          <motion.div
            key={index}
            className={`p-4 rounded-lg max-w-xs text-left shadow-lg ${
              msg.sender === "user" ? "bg-purple-500 text-white self-end ml-auto" : "bg-gray-700 text-gray-200 self-start"
            }`}
            initial={{ opacity: 0, x: msg.sender === "user" ? 50 : -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            {msg.text}
          </motion.div>
        ))}
        
        {loading && (
          <motion.div
            className="p-4 rounded-lg bg-gray-700 text-gray-200 self-start max-w-xs text-left shadow-lg"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Input Box */}
      <div className="flex space-x-3 w-full max-w-2xl">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          className="flex-1 p-3 rounded-xl bg-gray-800 text-white border border-gray-500 shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          placeholder="Ask a question..."
          disabled={loading}
        />
        <motion.button
          onClick={handleSendMessage}
          disabled={loading || !input.trim()}
          className={`px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 hover:shadow-2xl transition-all ${
            loading || !input.trim() ? "opacity-70 cursor-not-allowed" : ""
          }`}
          whileHover={{ scale: loading || !input.trim() ? 1 : 1.1 }}
          whileTap={{ scale: loading || !input.trim() ? 1 : 0.95 }}
        >
          Send
        </motion.button>
      </div>
    </div>
  );
}
