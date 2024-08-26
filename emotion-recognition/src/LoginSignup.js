import React, { useState } from "react";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { useNavigate } from "react-router-dom";
import { db } from "./firebase";
import { doc, setDoc, getDoc } from "firebase/firestore";
import "./LoginSignup.css";

const LoginSignup = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();
  const auth = getAuth();

  const handleAuth = async (e) => {
    e.preventDefault();

    try {
      if (isLogin) {
        // Login
        const userCredential = await signInWithEmailAndPassword(
          auth,
          email,
          password
        );
        const user = userCredential.user;

        // Get user role
        const userDoc = await getDoc(doc(db, "users", user.uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          const userRole = userData.role;

          if (userRole === "admin") {
            navigate("/admin");
          } else {
            navigate("/");
          }
        } else {
          console.error("No such user document!");
          alert("No such user document!");
        }
      } else {
        // Register
        const userCredential = await createUserWithEmailAndPassword(
          auth,
          email,
          password
        );
        const user = userCredential.user;

        // Set user role
        await setDoc(doc(db, "users", user.uid), {
          username,
          email,
          role: "student", // Default role
        });

        navigate("/dashboard");
      }
    } catch (error) {
      console.error("Error during authentication: ", error);
      alert("Authentication failed. Please check your credentials.");
    }
  };

  return (
    <div className="login-container">
      <div className="logo-section">
        <h1 className="logo">MoodMentor</h1>
      </div>
      <div className="form-section">
        <div className="form-container">
          <h2 className="welcome-text">Welcome to MoodMentor</h2>
          <div className="toggle-buttons">
            <button
              className={`toggle-btn ${isLogin ? "active" : ""}`}
              onClick={() => setIsLogin(true)}
            >
              Login
            </button>
            <button
              className={`toggle-btn ${!isLogin ? "active" : ""}`}
              onClick={() => setIsLogin(false)}
            >
              Register
            </button>
          </div>
          <p className="description">
            Unlock your full potential with personalized learning and emotional
            support!
          </p>
          <form onSubmit={handleAuth}>
            <div className="input-group">
              <label htmlFor="email">Email Address</label>
              <input
                type="email"
                id="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            {!isLogin && (
              <div className="input-group">
                <label htmlFor="username">User Name</label>
                <input
                  type="text"
                  id="username"
                  placeholder="Enter your user name"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
            )}
            <div className="input-group">
              <label htmlFor="password">Password</label>
              <div className="password-input">
                <input
                  type={showPassword ? "text" : "password"}
                  id="password"
                  placeholder="Enter your Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  className="toggle-password"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? "👁️" : "👁️‍🗨️"}
                </button>
              </div>
            </div>
            {isLogin && (
              <div className="options">
                <label className="remember-me">
                  <input type="checkbox" />
                  Remember me
                </label>
                <button type="button" className="forgot-password">
                  Forgot Password?
                </button>
              </div>
            )}
            <button type="submit" className="submit-btn">
              {isLogin ? "Login" : "Register"}
            </button>
          </form>
          <p className="terms">
            By creating an account, you have agreed to our{" "}
            <button type="button" className="terms-button">
              terms of service
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginSignup;
