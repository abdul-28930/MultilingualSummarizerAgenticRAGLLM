import BackgroundCanvas from "../components/BackgroundCanvas";

export default function Home() {
  return (
    <div className="relative flex items-center justify-center min-h-screen text-white">
      {/* Animated Background (Covers Full Viewport) */}
      <BackgroundCanvas />

      {/* Main Content */}
      <h1 className="text-4xl font-bold drop-shadow-lg">Welcome to Home Page</h1>
    </div>
  );
}
