import { motion } from "framer-motion";

export default function Navbar() {
  return (
    <motion.nav
      className="fixed top-0 left-0 w-full bg-gray-800 bg-opacity-90 shadow-md py-4 px-6 flex justify-between items-center z-50"
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Logo */}
      <motion.div
        className="text-xl font-bold text-blue-400"
        whileHover={{ scale: 1.1 }}
      >
        Multilingual Summarizer 🌍📜
      </motion.div>

      {/* Navigation Links */}
      <ul className="flex space-x-6 text-white font-semibold">
        <li>
          <a href="#hero" className="hover:text-blue-400 transition">Home</a>
        </li>
        <li>
          <a href="#features" className="hover:text-blue-400 transition">Features</a>
        </li>
        <li>
          <a href="#about" className="hover:text-blue-400 transition">About</a>
        </li>
      </ul>

      {/* Get Started Button */}
      <motion.button
        className="px-4 py-2 bg-blue-500 text-white rounded-lg shadow-lg hover:bg-blue-600 transition"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        Get Started
      </motion.button>
    </motion.nav>
  );
}
