// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyD-HMuTFks-efPhGrWiyc2YDL6m4dpMLcw",
  authDomain: "emotion-app-86b8c.firebaseapp.com",
  databaseURL: "https://emotion-app-86b8c-default-rtdb.firebaseio.com",
  projectId: "emotion-app-86b8c",
  storageBucket: "emotion-app-86b8c.appspot.com",
  messagingSenderId: "806450036833",
  appId: "1:806450036833:web:31ae018e0eb44b9147aa20",
  measurementId: "G-YKGR7WMGP2",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

const auth = getAuth(app);
const db = getFirestore(app);

export { auth, db };
