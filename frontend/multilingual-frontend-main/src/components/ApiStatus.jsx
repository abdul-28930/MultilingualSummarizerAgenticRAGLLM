import { useState, useEffect } from "react";
import { checkApiHealth } from "../services/api";

export default function ApiStatus() {
  const [isApiConnected, setIsApiConnected] = useState(null);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const isConnected = await checkApiHealth();
        setIsApiConnected(isConnected);
      } catch (error) {
        setIsApiConnected(false);
      }
    };

    checkConnection();
    
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (isApiConnected === null) {
    return (
      <div className="fixed bottom-4 right-4 px-4 py-2 bg-gray-800 text-white rounded-lg shadow-lg">
        Checking API connection...
      </div>
    );
  }

  if (!isApiConnected) {
    return (
      <div className="fixed bottom-4 right-4 px-4 py-2 bg-red-800 text-white rounded-lg shadow-lg">
        ⚠️ API not connected. Please start the backend server.
      </div>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 px-4 py-2 bg-green-800 text-white rounded-lg shadow-lg opacity-80 hover:opacity-100 transition-opacity">
      ✅ API Connected
    </div>
  );
}
