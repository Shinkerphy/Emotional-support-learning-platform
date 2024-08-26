import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Routes,
  Navigate,
} from "react-router-dom";
import Home from "./Home";
import EmotionDetection from "./EmotionDetection";
import Dashboard from "./Dashboard";
import Lesson from "./Lesson";
import ModuleContent from "./ModuleContent"; // Import the new component
import AdminDashboard from "./AdminDashboard";
import LoginSignup from "./LoginSignup";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { getDoc, doc } from "firebase/firestore";
import { db } from "./firebase";

const PrivateRoute = ({ children, role }) => {
  const [authenticated, setAuthenticated] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [userRole, setUserRole] = React.useState(null);

  React.useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        const userDoc = await getDoc(doc(db, "users", user.uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          setUserRole(userData.role);
          setAuthenticated(true);
        } else {
          console.error("No such user document!");
        }
      } else {
        setAuthenticated(false);
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!authenticated) {
    return <Navigate to="/login" />;
  }

  if (role && userRole !== role) {
    return <Navigate to="/" />;
  }

  return children;
};

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/login" element={<LoginSignup />} />
          <Route
            path="/"
            element={
              <PrivateRoute>
                <Home />
              </PrivateRoute>
            }
          />
          <Route
            path="/emotion-detection"
            element={
              <PrivateRoute>
                <EmotionDetection />
              </PrivateRoute>
            }
          />
          <Route
            path="/lesson"
            element={
              <PrivateRoute>
                <Lesson />
              </PrivateRoute>
            }
          />
          <Route
            path="/dashboard"
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/admin"
            element={
              <PrivateRoute role="admin">
                <AdminDashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/modules/:moduleId"
            element={
              <PrivateRoute>
                <ModuleContent />
              </PrivateRoute>
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
