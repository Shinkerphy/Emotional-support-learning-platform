import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { ChevronLeft } from "lucide-react";
import { Link } from "react-router-dom";
import { PieChart, Pie, Cell } from "recharts";
import { auth, db } from "./firebase";
import {
  collection,
  query,
  where,
  getDocs,
  doc,
  getDoc,
} from "firebase/firestore";
import "./Dashboard.css";

const emotionColors = {
  Angry: "#FF6B6B",
  Contempt: "#FFB347",
  Disgusted: "#4ECDC4",
  Fearful: "#45B7D1",
  Happy: "#FFA07A",
  Neutral: "#98D8C8",
  Sad: "#F06292",
  Surprised: "#AED581",
  "No faces detected": "#B0BEC5",
};

const Dashboard = () => {
  const [activeView, setActiveView] = useState("studentList");
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [students, setStudents] = useState([]);
  const [emotionDataByDate, setEmotionDataByDate] = useState({});
  const [allEmotions, setAllEmotions] = useState([]);
  const [expandedDates, setExpandedDates] = useState([]);
  const [studentEmotionData, setStudentEmotionData] = useState([]);

  useEffect(() => {
    const fetchStudents = async () => {
      try {
        const user = auth.currentUser;
        if (!user) return;

        const userDoc = await getDoc(doc(db, "users", user.uid));
        const userData = userDoc.data();
        if (userData.role !== "tutor") return;

        const coursesQuery = query(
          collection(db, "courses"),
          where("modules", "array-contains", userData.module)
        );
        const coursesSnapshot = await getDocs(coursesQuery);
        const courseNames = coursesSnapshot.docs.map((doc) => doc.data().name);

        const studentsQuery = query(
          collection(db, "users"),
          where("role", "==", "student"),
          where("course", "in", courseNames)
        );
        const studentsSnapshot = await getDocs(studentsQuery);
        const studentList = studentsSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setStudents(studentList);
      } catch (error) {
        console.error("Error fetching students: ", error);
      }
    };

    fetchStudents();
  }, []);

  useEffect(() => {
    const fetchStudentEmotions = async () => {
      if (selectedStudent) {
        try {
          const emotionsQuery = query(
            collection(db, "emotions"),
            where("studentId", "==", selectedStudent.id)
          );
          const emotionsSnapshot = await getDocs(emotionsQuery);
          const emotionsList = emotionsSnapshot.docs.map((doc) => ({
            ...doc.data(),
            timestamp: doc.data().timestamp.toDate(),
          }));

          const groupedByDate = emotionsList.reduce((acc, curr) => {
            const date = curr.timestamp.toLocaleDateString();
            if (!acc[date]) acc[date] = [];
            acc[date].push(curr);
            return acc;
          }, {});

          setEmotionDataByDate(groupedByDate);

          const emotionCounts = emotionsList.reduce((acc, curr) => {
            acc[curr.emotion] = (acc[curr.emotion] || 0) + 1;
            return acc;
          }, {});
          const emotionData = Object.keys(emotionCounts).map((emotion) => ({
            name: emotion,
            value: emotionCounts[emotion],
          }));
          setStudentEmotionData(emotionData);
        } catch (error) {
          console.error("Error fetching student emotions: ", error);
        }
      }
    };

    fetchStudentEmotions();
  }, [selectedStudent]);

  useEffect(() => {
    const fetchAllEmotions = async () => {
      try {
        const emotionsQuery = query(collection(db, "emotions"));
        const emotionsSnapshot = await getDocs(emotionsQuery);
        const emotionsList = emotionsSnapshot.docs.map((doc) => ({
          ...doc.data(),
          timestamp: doc.data().timestamp.toDate(),
        }));
        setAllEmotions(emotionsList);
      } catch (error) {
        console.error("Error fetching all emotions: ", error);
      }
    };

    fetchAllEmotions();
  }, []);

  const calculateCumulativeEmotions = () => {
    const emotionCounts = allEmotions.reduce((acc, curr) => {
      acc[curr.emotion] = (acc[curr.emotion] || 0) + 1;
      return acc;
    }, {});

    const emotionData = Object.keys(emotionCounts).map((emotion) => ({
      name: emotion,
      value: emotionCounts[emotion],
    }));

    const mostCommonEmotion = Object.keys(emotionCounts).reduce((a, b) =>
      emotionCounts[a] > emotionCounts[b] ? a : b
    );

    return { emotionData, mostCommonEmotion };
  };

  const toggleDateExpand = (date) => {
    setExpandedDates((prevExpandedDates) =>
      prevExpandedDates.includes(date)
        ? prevExpandedDates.filter((d) => d !== date)
        : [...prevExpandedDates, date]
    );
  };

  const StudentList = () => (
    <div className="student-list">
      {students.map((student) => (
        <div key={student.id} className="student-item">
          <span>{student.name}</span>
          <button
            onClick={() => {
              setSelectedStudent(student);
              setActiveView("emotionsInfo");
            }}
          >
            &gt;
          </button>
        </div>
      ))}
    </div>
  );

  const EmotionsInfo = () => (
    <div className="emotions-info">
      <div className="student-header">
        <img
          src="https://picsum.photos/24/24"
          alt={selectedStudent?.name}
          className="student-avatar"
        />
        <span className="student-name">{selectedStudent?.name}</span>
      </div>
      <div className="data-section">
        {Object.keys(emotionDataByDate).map((date) => (
          <div key={date} className="date-section">
            <div className="date-header" onClick={() => toggleDateExpand(date)}>
              <h3>{date}</h3>
              <button className="toggle-button">
                {expandedDates.includes(date) ? "Hide" : "Show"} Emotions
              </button>
            </div>
            {expandedDates.includes(date) && (
              <div className="emotions-record">
                <ul>
                  {emotionDataByDate[date].map((entry, index) => (
                    <li key={index}>
                      Time: {entry.timestamp.toLocaleTimeString()} - Emotion:{" "}
                      {entry.emotion}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="chart-container">
        <h3>{selectedStudent?.name}'s Emotion Data</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={studentEmotionData}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {/* Single Bar with distinct colors for each emotion */}
            <Bar dataKey="value" name="Emotions" label={{ position: "top" }}>
              {studentEmotionData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={emotionColors[entry.name]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="pie-chart-container">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={studentEmotionData}
                cx="50%"
                cy="50%"
                outerRadius={70}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                {studentEmotionData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={emotionColors[entry.name]}
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <button
        className="back-button"
        onClick={() => setActiveView("studentList")}
      >
        Back to Student List
      </button>
    </div>
  );

  const AllUsersEmotions = () => {
    const { emotionData, mostCommonEmotion } = calculateCumulativeEmotions();

    return (
      <div className="all-users-emotions">
        <div className="module-header">
          <img
            src="https://picsum.photos/24/24"
            alt="Deep Learning Module"
            className="module-avatar"
          />
          <span className="module-name">Deep Learning Module</span>
        </div>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={emotionData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              {/* Here we use one single Bar, and dynamically set the color based on the name */}
              <Bar
                dataKey="value"
                name="Emotions"
                label={{ position: "top" }} // Show value on top of bars
              >
                {emotionData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={emotionColors[entry.name]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="pie-chart-container">
          <h3>Cumulative Emotion Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={emotionData}
                cx="50%"
                cy="50%"
                className="PieChart"
                outerRadius={70}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                {emotionData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={emotionColors[entry.name]}
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="emotion-summary">
          <h3>Emotion Insight</h3>
          <p>
            The most common emotion among students in this module is{" "}
            <strong>{mostCommonEmotion}</strong>.
          </p>
        </div>

        <button
          className="back-button"
          onClick={() => setActiveView("studentList")}
        >
          Back to Student List
        </button>
      </div>
    );
  };

  return (
    <div className="dashboard-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <Link to="/" className="back-button">
            <ChevronLeft size={24} /> Home
          </Link>
          <h1 className="sidebar-title">Users Dashboard</h1>
        </div>
        <div className="sidebar-content">
          <h3>Students</h3>
          <ul className="sidebar-nav">
            <li>
              <button
                className={activeView === "studentList" ? "active" : ""}
                onClick={() => setActiveView("studentList")}
              >
                Student List
              </button>
            </li>
            <li>
              <button
                className={activeView === "emotionsInfo" ? "active" : ""}
                onClick={() => setActiveView("emotionsInfo")}
              >
                Student's Dashboard
              </button>
            </li>
            <li>
              <button
                className={activeView === "allUsersEmotions" ? "active" : ""}
                onClick={() => setActiveView("allUsersEmotions")}
              >
                Emotion's Dashboard
              </button>
            </li>
          </ul>
        </div>
      </div>
      <div className="main-content">
        {activeView === "studentList" && <StudentList />}
        {activeView === "emotionsInfo" && <EmotionsInfo />}
        {activeView === "allUsersEmotions" && <AllUsersEmotions />}
      </div>
      <div className="footer">
        <span className="copyright">
          © 2024 MoodMentor. All rights reserved.
        </span>
      </div>
    </div>
  );
};

export default Dashboard;
