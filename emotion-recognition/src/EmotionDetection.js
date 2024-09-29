import React, { useEffect, useRef, useState } from "react";
import { ChevronLeft, Settings } from "lucide-react";
import { Link } from "react-router-dom";
import "./EmotionDetection.css"; // Ensure this file is created for additional styling

function EmotionDetection() {
  const videoRef = useRef(null);
  const [emotion, setEmotion] = useState("");
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.src = "http://localhost:5001/video_feed";
    }

    const fetchEmotion = async () => {
      if (isDetectionActive) {
        try {
          const response = await fetch("http://localhost:5001/current_emotion");
          const data = await response.json();
          setEmotion(data.emotion);
        } catch (error) {
          console.error("Error fetching emotion:", error);
        }
      }
    };

    const interval = setInterval(fetchEmotion, 1000); // Fetch emotion every second

    return () => clearInterval(interval);
  }, [isDetectionActive]);

  useEffect(() => {
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timeInterval);
  }, []);

  const getMotivationalMessage = (emotion) => {
    const messages = {
      Angry:
        "I know it can be frustrating 😤. Take a deep breath, relax, and keep pushing forward! 💪",
      Disgusted:
        "It's natural to feel put off sometimes 😖. Take a break and refocus when you're ready 🌱.",
      Fearful:
        "It's normal to feel uncertain 😨, but you're doing great! Stay confident and trust yourself 🌟.",
      Happy: "Awesome! 😄 Keep up the amazing work! You're on fire! 🔥",
      Neutral:
        "You're doing well 👍. Stay focused and continue at your pace 🧠.",
      Sad: "It's okay to feel down sometimes 😔. Take a break, be kind to yourself, and come back stronger 💜.",
      Surprised:
        "Surprises can throw us off 😲, but you've got this! Keep going and stay flexible.",
    };
    return messages[emotion] || "Detecting emotion...";
  };

  const formatDateTime = (date) => {
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const seconds = String(date.getSeconds()).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0"); // Months are zero-based
    const year = date.getFullYear();
    return `${hours}:${minutes}:${seconds} ${day}/${month}/${year}`;
  };

  return (
    <div className="emotion-detection">
      <div className="header">
        <Link to="/" className="back-button">
          <ChevronLeft size={24} />
        </Link>
        <h1>Real-time Emotion Detection</h1>
        <div className="header-info">
          <span>Abdul</span>
          <span>{formatDateTime(currentTime)}</span>
        </div>
        <Settings className="settings-icon" size={24} />
      </div>
      <div className="content-row">
        <div className="video-container">
          <div className="video-wrapper">
            <img ref={videoRef} alt="Real-time video feed" />
          </div>
          <div className="controls">
            <button
              className="control-button activate"
              onClick={() => setIsDetectionActive(true)}
            >
              Activate Detection
            </button>
            <button
              className="control-button deactivate"
              onClick={() => setIsDetectionActive(false)}
            >
              Deactivate Detection
            </button>
          </div>
        </div>
        <div className="emotion-info">
          <h2>Detected Emotion</h2>
          <div className="emotion-details">
            <div className="emotion-time">
              <span>🕒 {formatDateTime(currentTime)}</span>
              <span>{emotion || "Happiness"}</span>
            </div>
            <div className="message-box">
              <p>
                {emotion
                  ? getMotivationalMessage(emotion)
                  : "Great job! Your positive attitude is inspiring. Keep up the excellent work!"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default EmotionDetection;
