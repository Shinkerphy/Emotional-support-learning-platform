import React, { useState, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Search,
  MessageCircle,
  UserCheck,
  Smile,
  Monitor,
  Mail,
  Phone,
  MapPin,
  LogOut,
} from "lucide-react";
import "./Home.css";
import { Link, useNavigate } from "react-router-dom";
import { getAuth, signOut, onAuthStateChanged } from "firebase/auth";
import {
  query,
  collection,
  where,
  getDocs,
  getDoc,
  doc,
} from "firebase/firestore";
import { db } from "./firebase";

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [userRole, setUserRole] = useState(null);
  const navigate = useNavigate();
  const auth = getAuth();

  useEffect(() => {
    onAuthStateChanged(auth, async (user) => {
      if (user) {
        const userDoc = await getDoc(doc(db, "users", user.uid));
        if (userDoc.exists()) {
          setUserRole(userDoc.data().role);
        }
      }
    });
  }, [auth]);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate("/login");
    } catch (e) {
      console.error("Error logging out: ", e);
    }
  };

  return (
    <header className="header">
      <div className="logo">MoodMentor</div>
      <nav className={`nav ${isMobileMenuOpen ? "open" : ""}`}>
        <a href="/" className="nav-link">
          Home
        </a>
        <Link to="/lesson" className="nav-link">
          Lesson
        </Link>
        <Link to="/emotion-detection" className="nav-link">
          Emotions
        </Link>
        {userRole === "tutor" && (
          <Link to="/dashboard" className="nav-link">
            Dashboard
          </Link>
        )}
      </nav>
      <div className="user-profile">
        <img
          src="https://picsum.photos/24/24"
          alt="User"
          className="user-avatar"
        />
        <span className="username">Abdul</span>
        <button className="logout-button" onClick={handleLogout}>
          <LogOut size={16} />
        </button>
      </div>
      <button className="mobile-menu-button" onClick={toggleMobileMenu}>
        {isMobileMenuOpen ? <X /> : <Menu />}
      </button>
    </header>
  );
};

const SearchBar = () => (
  <div className="search-container">
    <input
      type="text"
      placeholder="Search your favourite course"
      className="search-input"
    />
    <button className="search-button">
      <Search size={20} />
    </button>
  </div>
);

const CourseCard = ({
  title,
  instructor,
  progress,
  image,
  moduleId,
  label,
}) => (
  <Link to={`/modules/${moduleId}`} className="course-card">
    <img src={image} alt={title} className="course-image" />
    {label && <div className="red-label">{label}</div>}
    <div className="course-info">
      <h3>{title}</h3>
      <div className="instructor">
        <img
          src="https://picsum.photos/24/24"
          alt={instructor}
          className="instructor-avatar"
        />
        <span>{instructor}</span>
      </div>
    </div>
  </Link>
);

const CoursesSection = () => {
  const [modules, setModules] = useState([]);
  const auth = getAuth();
  const user = auth.currentUser;

  useEffect(() => {
    const fetchUserModules = async () => {
      if (!user) return;

      try {
        const userDoc = await getDoc(doc(db, "users", user.uid));
        const userData = userDoc.data();

        if (userData.role === "student") {
          const courseRef = query(
            collection(db, "courses"),
            where("name", "==", userData.course)
          );
          const courseSnapshot = await getDocs(courseRef);
          if (!courseSnapshot.empty) {
            const courseDoc = courseSnapshot.docs[0];
            const courseData = courseDoc.data();
            const modulesData = await Promise.all(
              courseData.modules.map(async (moduleName) => {
                const moduleRef = query(
                  collection(db, "modules"),
                  where("name", "==", moduleName)
                );
                const moduleSnapshot = await getDocs(moduleRef);
                return moduleSnapshot.docs[0];
              })
            );
            setModules(modulesData);
          }
        } else if (userData.role === "tutor") {
          const moduleRef = query(
            collection(db, "modules"),
            where("name", "==", userData.module)
          );
          const moduleSnapshot = await getDocs(moduleRef);
          setModules(moduleSnapshot.docs);
        }
      } catch (error) {
        console.error("Error fetching user modules: ", error);
      }
    };

    fetchUserModules();
  }, [user]);

  return (
    <section className="courses-section">
      <h2 className="section-heading">
        Welcome back, ready for your next lesson?
      </h2>
      <div className="courses-grid">
        {modules.length > 0 ? (
          modules.map((module, index) => (
            <CourseCard
              key={index}
              title={module.data().name}
              instructor="Lina"
              image={`https://picsum.photos/300/200?random=${index}`}
              moduleId={module.id}
              label="Mood Mentor Academy"
            />
          ))
        ) : (
          <CourseCard
            title="Sample Module"
            instructor="Lina"
            image="https://picsum.photos/300/200?random=1"
            label="Support"
          />
        )}
      </div>
      <div className="pagination">
        <button className="page-button">
          <ChevronLeft />
        </button>
        <button className="page-button">
          <ChevronRight />
        </button>
      </div>
    </section>
  );
};

const FeaturesSection = () => (
  <section className="features-section">
    <h2 className="features-heading">What makes MoodMentor Different?</h2>
    <div className="features-grid">
      <div className="feature-card">
        <MessageCircle size={48} color="var(--primary-blue)" />
        <h3>Realtime Support</h3>
        <p>
          Receive instant feedback and emotional insights to help you understand
          your learning patterns and improve effectively.
        </p>
      </div>
      <div className="feature-card">
        <UserCheck size={48} color="var(--primary-blue)" />
        <h3>Personalized Learning</h3>
        <p>
          Our advanced technology adapts to your emotional state, providing
          personalized support and motivation to enhance your learning journey.
        </p>
      </div>
      <div className="feature-card">
        <Smile size={48} color="var(--primary-blue)" />
        <h3>Emotion-Aware</h3>
        <p>
          Harness the power of emotion recognition to tailor your learning
          experience. We adapt to your emotional state, offering support.
        </p>
      </div>
      <div className="feature-card">
        <Monitor size={48} color="var(--primary-blue)" />
        <h3>Interactive Courses</h3>
        <p>
          Engage with interactive and immersive course materials designed to
          keep you motivated and excited about learning.
        </p>
      </div>
    </div>
  </section>
);

const EmpoweringSection = () => (
  <section className="empowering-section">
    <h2 className="empowering-heading">
      Empowering Education with Emotional Intelligence
    </h2>
    <p>
      Unlock a new way of learning with MoodMentor. Our platform combines
      cutting-edge educational resources with advanced emotion recognition
      technology to create a personalized and supportive learning experience.
      Whether you're a student seeking to enhance your studies or a tutor aiming
      to understand your students better, MoodMentor is here to help you
      succeed.
    </p>
    <button className="join-button">Join us Today</button>
  </section>
);

const Footer = () => (
  <footer className="home-footer">
    <div className="home-footer-content">
      <div className="home-footer-logo">
        <h2>MoodMentor</h2>
        <p>
          Unlock your full potential with personalized learning and emotional
          support.
        </p>
      </div>
      <div className="home-footer-links">
        <div className="home-footer-column">
          <h3>Quick Links</h3>
          <ul>
            <li>
              <a href="/detect-emotions">Detect Emotions</a>
            </li>
            <li>
              <a href="/courses">Courses</a>
            </li>
            <li>
              <a href="/about-us">About Us</a>
            </li>
          </ul>
        </div>
        <div className="home-footer-column">
          <h3>Contact Us</h3>
          <ul>
            <li>
              <Mail size={16} /> <span>Email: support@MoodMentor.com</span>
            </li>
            <li>
              <Phone size={16} /> <span>Phone: +1 (123) 456-7890</span>
            </li>
            <li>
              <MapPin size={16} />{" "}
              <span>Address: Northampton Square, EC1 258</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div className="home-footer-bottom">
      <nav className="home-footer-nav">
        <a href="/courses">Courses</a>
        <a href="/privacy-policy">Privacy Policy</a>
        <a href="/terms">Terms & Conditions</a>
      </nav>
      <p>&copy; 2024 MoodMentor. All rights reserved.</p>
    </div>
  </footer>
);

const Home = () => (
  <div className="home">
    <Header />
    <main>
      <SearchBar />
      <CoursesSection />
      <FeaturesSection />
      <EmpoweringSection />
    </main>
    <Footer />
  </div>
);

export default Home;
