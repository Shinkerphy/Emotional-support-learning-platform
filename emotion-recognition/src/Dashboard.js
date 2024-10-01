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
  const [emotionData, setEmotionData] = useState([]);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [allEmotions, setAllEmotions] = useState([]);
  const [showMoreEmotions, setShowMoreEmotions] = useState(false);

  useEffect(() => {
    const fetchStudents = async () => {
      try {
        const user = auth.currentUser;
        if (!user) return;

        const userDoc = await getDoc(doc(db, "users", user.uid));
        const userData = userDoc.data();
        if (userData.role !== "tutor") return;

        // Get all courses that include the tutor's module
        const coursesQuery = query(
          collection(db, "courses"),
          where("modules", "array-contains", userData.module)
        );
        const coursesSnapshot = await getDocs(coursesQuery);
        const courseNames = coursesSnapshot.docs.map((doc) => doc.data().name);

        // Get all students who are taking any of the courses
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

          // Aggregate emotion data for the pie chart
          const emotionCounts = emotionsList.reduce((acc, curr) => {
            acc[curr.emotion] = (acc[curr.emotion] || 0) + 1;
            return acc;
          }, {});

          const emotionData = Object.keys(emotionCounts).map((emotion) => ({
            name: emotion,
            value: emotionCounts[emotion],
          }));
          setEmotionData(emotionData);

          // Prepare time series data for the bar chart
          const timeSeriesData = emotionsList.reduce((acc, curr) => {
            const timeKey = curr.timestamp.toLocaleTimeString();
            if (!acc[timeKey]) {
              acc[timeKey] = { time: timeKey };
            }
            acc[timeKey][curr.emotion] = (acc[timeKey][curr.emotion] || 0) + 1;
            return acc;
          }, {});

          setTimeSeriesData(Object.values(timeSeriesData));
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

  // Helper function to calculate average emotions for all students
  const calculateAverageEmotion = () => {
    const emotionCounts = allEmotions.reduce((acc, curr) => {
      acc[curr.emotion] = (acc[curr.emotion] || 0) + 1;
      return acc;
    }, {});

    // Removed unused dominantEmotion and directly generate summary
    const summary = `Most students in this module are feeling ${Object.keys(
      emotionCounts
    ).reduce((a, b) => (emotionCounts[a] > emotionCounts[b] ? a : b))}.`;

    return { summary };
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
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={timeSeriesData}>
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            {Object.keys(emotionColors).map((emotion) => (
              <Bar
                key={emotion}
                dataKey={emotion}
                stackId="a"
                fill={emotionColors[emotion]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="data-section">
        <div className="pie-chart-container">
          <h3>Emotion Distribution Analysis</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={emotionData}
                cx="50%"
                cy="50%"
                outerRadius={80}
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
        <div className="emotions-record">
          <h3>Emotions Record</h3>
          <table>
            <tbody>
              {timeSeriesData.slice(0, 15).map((entry, index) => (
                <tr key={index}>
                  <td>{entry.time}</td>
                  <td>
                    {Object.keys(entry)
                      .filter((key) => key !== "time")
                      .join(", ")}
                  </td>
                </tr>
              ))}
              {timeSeriesData.length > 15 && (
                <tr>
                  <td colSpan="2">
                    <button
                      onClick={() => setShowMoreEmotions(!showMoreEmotions)}
                    >
                      {showMoreEmotions ? "Show Less" : "Show More"}
                    </button>
                  </td>
                </tr>
              )}
              {showMoreEmotions &&
                timeSeriesData.slice(15).map((entry, index) => (
                  <tr key={index}>
                    <td>{entry.time}</td>
                    <td>
                      {Object.keys(entry)
                        .filter((key) => key !== "time")
                        .join(", ")}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
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
    const { summary } = calculateAverageEmotion(); // Use only summary

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
            <BarChart data={timeSeriesData}>
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(emotionColors).map((emotion) => (
                <Bar
                  key={emotion}
                  dataKey={emotion}
                  stackId="a"
                  fill={emotionColors[emotion]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="emotion-summary">
          <h3>Emotion Insight</h3>
          <p>{summary}</p> {/* Displaying the emotion insight summary */}
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
