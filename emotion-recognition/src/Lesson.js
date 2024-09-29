import React, { useEffect, useRef, useState } from "react";
import "./Lesson.css";
import { useNavigate } from "react-router-dom";
import Modal from "react-modal";

Modal.setAppElement("#root");

const Lesson = () => {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const [emotion, setEmotion] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDetectionActive, setIsDetectionActive] = useState(false);

  const goBack = () => {
    navigate(-1); // This will navigate back to the previous page
  };

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
    <div className="lesson">
      <header className="course-header">
        <button className="back-button" onClick={goBack}>
          &#8592; Back
        </button>
        <div className="course-name">
          Generative Adversarial Networks (GANS)
        </div>
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
        <h1>Generative Adversarial Networks (GANS)</h1>
        <p>
          Generative Adversarial Networks (GANs) are a class of machine learning
          frameworks designed by Ian Goodfellow and his colleagues in 2014. They
          consist of two neural networks: a generator and a discriminator, which
          are trained simultaneously through a process of adversarial learning.
        </p>
        <ol>
          <li>
            Generator: This network creates synthetic data (e.g., images) from
            random noise. Its goal is to produce data that is indistinguishable
            from real data.
          </li>
          <li>
            Discriminator: This network evaluates the data, distinguishing
            between real and generated (fake) data. Its goal is to correctly
            identify which data is real and which is generated.
          </li>
        </ol>
        <div className="diagram">
          <img
            src="https://via.placeholder.com/1200x400.png?text=Meeting+Image"
            alt="GAN Diagram"
          />
        </div>
        <p>
          The training process involves the generator and discriminator in a
          competitive game:
        </p>
        <ul>
          <li>
            The generator tries to create more realistic data to fool the
            discriminator.
          </li>
          <li>
            The discriminator tries to get better at identifying fake data.
          </li>
        </ul>
        <p>
          Over time, both networks improve, resulting in the generator producing
          highly realistic data. GANs have been widely used in various
          applications, such as image and video generation, data augmentation,
          and even in generating music and text. Their ability to create
          realistic synthetic data makes them powerful tools in both research
          and industry.
        </p>

        <h2>Types of GANS</h2>

        <div className="gans">
          <div className="gan">
            <h3>DCGANS</h3>
            <h4>
              Deep Convolutional Generative Adversarial Neural Network (DCGAN)
            </h4>
            <div className="gan-content">
              <p>
                A Deep Convolutional Generative Adversarial Network (DCGAN) is a
                type of Generative Adversarial Network (GAN) that uses deep
                convolutional layers to generate realistic images. It consists
                of two main components, the Generator and Discriminator.
              </p>
              <div className="gan-image">
                <img
                  src="https://via.placeholder.com/300x200.png?text=DCGAN+Image"
                  alt="DCGAN"
                />
              </div>
            </div>
          </div>

          <div className="gan">
            <h3>ProGANS</h3>
            <h4>Progressive Generative Adversarial Neural Network (ProGAN)</h4>
            <div className="gan-content">
              <p>
                Progressive Growing of GANs (ProGAN) is an advanced type of
                Generative Adversarial Network designed to generate
                high-resolution images with better quality and stability.
                Introduced by Tero Karras and colleagues in 2017, ProGAN
                improves upon traditional GANs by progressively increasing the
                size and complexity of both the generator and discriminator
                networks during training.
              </p>
              <div className="gan-image">
                <img
                  src="https://via.placeholder.com/300x200.png?text=ProGAN+Image"
                  alt="ProGAN"
                />
              </div>
            </div>
          </div>

          <div className="gan">
            <h3>StyleGAN</h3>
            <h4>StyleGAN (Style-based Generative Adversarial Network)</h4>
            <div className="gan-content">
              <p>
                StyleGAN is an advanced GAN introduced by NVIDIA researchers in
                2018. It enhances image generation quality and control through
                several key innovations:
              </p>
              <ul>
                <li>
                  Style-Based Generator: Injects style information at different
                  layers, allowing precise control over image features like
                  texture, color, and structure.
                </li>
                <li>
                  Adaptive Instance Normalization (AdaIN): Normalizes feature
                  maps and adjusts them with learned parameters, enabling
                  fine-grained style blending.
                </li>
                <li>
                  Progressive Growing: Gradually increases image resolution
                  during training, improving stability and quality.
                </li>
                <li>
                  Feature Separation: Controls coarse (pose, shape) and fine
                  details (texture, freckles) independently.
                </li>
              </ul>
              <div className="gan-image">
                <img
                  src="https://via.placeholder.com/300x200.png?text=StyleGAN+Image"
                  alt="StyleGAN"
                />
              </div>
            </div>
          </div>
          <h1>StyleGAN(SAGANS)</h1>
          <p>
            StyleGAN (Style-based Generative Adversarial Network) is an advanced
            GAN introduced by NVIDIA researchers in 2018. It enhances image
            generation quality and control through several key innovations:
          </p>
          <ol>
            <li>
              Style-Based Generator: Injects style information at different
              layers, allowing precise control over image features like texture,
              color, and structure
            </li>
            <li>
              Adaptive Instance Normalization (AdaIN): Normalizes feature maps
              and adjusts them with learned parameters, enabling fine-grained
              style blending.
            </li>
            <li>
              Progressive Growing: Gradually increases image resolution during
              training, improving stability and quality.
            </li>
            <li>
              Feature Separation: Controls coarse (pose, shape) and fine details
              (texture, freckles) independently.
            </li>
          </ol>
          <div className="diagram">
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/_qB4B6ttXk8?si=-GAXw7ux59X1S0aA"
              title="YouTube video player"
              frameborder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerpolicy="strict-origin-when-cross-origin"
              allowfullscreen
            ></iframe>
          </div>
          <p>
            StyleGAN produces highly realistic images and is used in
            applications like deepfake creation, virtual character design, and
            data augmentation. The improved StyleGAN2 version further enhances
            image quality and reduces artifacts.
          </p>
        </div>
      </section>
      <section className="education-sources">
        <h2>Top Education Sources</h2>
        <div className="sources">
          <div className="source">
            <h3>YouTube</h3>
            <p>
              TOTC's school management software helps traditional and online
              schools manage scheduling.
            </p>
          </div>
          <div className="source">
            <h3>Udemy</h3>
            <p>
              TOTC's school management software helps traditional and online
              schools manage scheduling.
            </p>
          </div>
          <div className="source">
            <h3>Coursera</h3>
            <p>
              TOTC's school management software helps traditional and online
              schools manage scheduling.
            </p>
          </div>
        </div>
      </section>

      <footer>
        <div className="footer-content">
          <div className="footer-logo">MoodMentor</div>
          <p>
            Unlock your full potential with personalized learning and emotional
            support. Log in to start your journey towards a more mindful and
            motivated learning experience.
          </p>
          <div className="contact">
            <h4>Contact Us</h4>
            <p>Email: support@MoodMentor.com</p>
            <p>Phone: +1 (123) 456-7890</p>
            <p>Address: Northampton Square, EC1 258</p>
          </div>
        </div>
      </footer>

      <Modal
        isOpen={isModalOpen}
        onRequestClose={() => setIsModalOpen(false)}
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

export default Lesson;
