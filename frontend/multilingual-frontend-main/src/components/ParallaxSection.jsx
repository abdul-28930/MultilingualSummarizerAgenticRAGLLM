import { Parallax } from "react-scroll-parallax";

export default function ParallaxSection() {
  return (
    <div className="relative flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
      {/* Parallax Heading */}
      <Parallax speed={-10}>
        <h1 className="text-5xl font-bold text-blue-400">
          Elevate Your Summarization! 🚀
        </h1>
      </Parallax>

      {/* Parallax Icons */}
      <Parallax speed={5}>
        <p className="mt-4 text-gray-300 text-lg max-w-lg text-center">
          Experience AI-powered summarization with real-time transcription,
          translation, and query-based summaries.
        </p>
      </Parallax>

      {/* Get Started Button */}
      <Parallax speed={10}>
        <button className="mt-6 px-6 py-3 bg-blue-500 text-white font-semibold rounded-lg shadow-lg hover:bg-blue-600 transition">
          Try It Now
        </button>
      </Parallax>
    </div>
  );
}
