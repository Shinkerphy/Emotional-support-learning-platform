import React, { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { doc, getDoc, collection, getDocs, addDoc } from "firebase/firestore";
import { db, auth } from "./firebase";
import "./ModuleContent.css";
import Modal from "react-modal";

Modal.setAppElement("#root");

const ModuleContent = () => {
  const { moduleId } = useParams();
  const navigate = useNavigate();
  const [moduleData, setModuleData] = useState(null);
  const [emotion, setEmotion] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [lastEmotion, setLastEmotion] = useState(null);
  const [emotionTimer, setEmotionTimer] = useState(0);
  const [lastModalCloseTime, setLastModalCloseTime] = useState(null);
  const [saveTimer, setSaveTimer] = useState(0);

  const goBack = () => {
    navigate(-1); // This will navigate back to the previous page
  };

  useEffect(() => {
    const fetchModuleData = async () => {
      try {
        const docRef = doc(db, "modules", moduleId);
        const docSnap = await getDoc(docRef);
        if (docSnap.exists()) {
          const contentCollection = collection(docRef, "content");
          const contentSnapshot = await getDocs(contentCollection);
          const contentList = contentSnapshot.docs.map((doc) => doc.data());
          setModuleData({
            ...docSnap.data(),
            contentList,
          });
        } else {
          console.error("No such document!");
        }
      } catch (error) {
        console.error("Error fetching module data:", error);
      }
    };

    fetchModuleData();
  }, [moduleId]);

  const saveEmotionToDatabase = useCallback(
    async (detectedEmotion) => {
      const user = auth.currentUser;
      if (!user) return;

      const emotionData = {
        studentId: user.uid,
        moduleId,
        timestamp: new Date(),
        emotion: detectedEmotion,
      };

      try {
        await addDoc(collection(db, "emotions"), emotionData);
      } catch (error) {
        console.error("Error saving emotion to database:", error);
      }
    },
    [moduleId]
  );

  useEffect(() => {
    if (isDetectionActive) {
      const fetchEmotion = async () => {
        try {
          const response = await fetch("http://localhost:5001/current_emotion");
          const data = await response.json();
          setEmotion(data.emotion);

          if (data.emotion) {
            if (data.emotion === lastEmotion) {
              setEmotionTimer((prev) => prev + 1);
              if (
                emotionTimer >= 5 &&
                (!lastModalCloseTime || new Date() - lastModalCloseTime > 10000)
              ) {
                setIsModalOpen(true);
                setLastModalCloseTime(null); // Reset last modal close time
              }
            } else {
              setLastEmotion(data.emotion);
              setEmotionTimer(1);
            }

            if (saveTimer >= 20) {
              saveEmotionToDatabase(data.emotion);
              setSaveTimer(0);
            } else {
              setSaveTimer((prev) => prev + 1);
            }
          }
        } catch (error) {
          console.error("Error fetching emotion:", error);
        }
      };

      const interval = setInterval(fetchEmotion, 1000); // Fetch emotion every second

      return () => clearInterval(interval);
    }
  }, [
    isDetectionActive,
    emotionTimer,
    saveTimer,
    lastEmotion,
    saveEmotionToDatabase,
    lastModalCloseTime,
  ]);

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

  const getEmojiForEmotion = (emotion) => {
    const emojis = {
      Angry: "😠",
      Disgusted: "😒",
      Fearful: "😨",
      Happy: "😃",
      Neutral: "😐",
      Sad: "😢",
      Surprised: "😲",
    };
    return emojis[emotion] || "🤔";
  };

  return (
    <div className="module-content">
      <header className="module-header">
        <button className="back-button" onClick={goBack}>
          &#8592; Back
        </button>
        <div className="module-name">{moduleData?.name || "Loading..."}</div>
      </header>

      <div className="hero-section">
        <img
          src="https://via.placeholder.com/1200x400.png?text=Meeting+Image"
          alt="Meeting"
        />
      </div>
      <div className="buttons">
        <button className="activate" onClick={() => setIsDetectionActive(true)}>
          Activate Emotion Detection
        </button>
        <button
          className="deactivate"
          onClick={() => setIsDetectionActive(false)}
        >
          Deactivate Emotion Detection
        </button>
      </div>

      <section className="content">
        {moduleData ? (
          moduleData.contentList.map((content, index) => (
            <div key={index}>
              <div
                dangerouslySetInnerHTML={{ __html: content.textContent }}
              ></div>
              {content.images &&
                content.images.map((image, imgIndex) => (
                  <div key={imgIndex} className="diagram">
                    <img
                      src={image}
                      alt={`Module content ${imgIndex + 1}`}
                      className="centered-image"
                    />
                  </div>
                ))}
              {content.videos &&
                content.videos.map((video, vidIndex) => (
                  <div key={vidIndex} className="diagram">
                    <iframe
                      width="560"
                      height="315"
                      src={video}
                      title={`Video ${vidIndex + 1}`}
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      className="centered-video"
                    ></iframe>
                  </div>
                ))}
            </div>
          ))
        ) : (
          <p>Loading module content...</p>
        )}
      </section>

      <Modal
        isOpen={isModalOpen}
        onRequestClose={() => {
          setIsModalOpen(false);
          setLastModalCloseTime(new Date());
        }}
        contentLabel="Motivational Message"
        className="Modal"
        overlayClassName="Overlay"
      >
        <div className="modal-content">
          <div className="emoji-wrapper">
            <span className="emoji">{getEmojiForEmotion(emotion)}</span>
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
};

export default ModuleContent;
