import React, { useEffect, useRef, useState } from "react";
import Modal from "react-modal";
import "./App.css"; // Ensure this file is created for additional styling

Modal.setAppElement("#root");

function Module() {
  const videoRef = useRef(null);
  const [emotion, setEmotion] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDetectionActive, setIsDetectionActive] = useState(false);

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
          if (data.emotion) {
            setIsModalOpen(true);
          }
        } catch (error) {
          console.error("Error fetching emotion:", error);
        }
      }
    };

    const interval = setInterval(fetchEmotion, 1000); // Fetch emotion every second

    return () => clearInterval(interval);
  }, [isDetectionActive]);

  const getMotivationalMessage = (emotion) => {
    const messages = {
      Angry: "Take a deep breath and relax. Keep pushing forward!",
      Disgusted: "It's okay to take a break. Try to refocus.",
      Fearful: "You're doing great. Stay confident!",
      Happy: "Awesome! Keep up the good work!",
      Neutral: "Stay focused and keep working.",
      Sad: "It's okay to feel down. Take a short break.",
      Surprised: "Unexpected things happen. Keep going!",
    };
    return messages[emotion] || "Detecting emotion...";
  };

  return (
    <div className="emotion-detection">
      <h1>Emotion Detection</h1>
      <div className="video-wrapper">
        <video ref={videoRef} width="640" height="480" autoPlay></video>
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
      <Modal
        isOpen={isModalOpen}
        onRequestClose={() => setIsModalOpen(false)}
        contentLabel="Motivational Message"
        className="Modal"
        overlayClassName="Overlay"
      >
        <div className="modal-content">
          <div className="video-wrapper">
            <video ref={videoRef} width="320" height="240" autoPlay></video>
          </div>
          <div className="message-content">
            <h2>Detected Emotion: {emotion}</h2>
            <p>{getMotivationalMessage(emotion)}</p>
            <button onClick={() => setIsModalOpen(false)}>Close</button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

export default Module;
