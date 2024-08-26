import React, { useState, useEffect } from "react";
import {
  collection,
  addDoc,
  getDocs,
  query,
  where,
  doc,
  setDoc,
  updateDoc,
  arrayUnion,
} from "firebase/firestore";
import { db, auth } from "./firebase";
import { useNavigate } from "react-router-dom";
import { createUserWithEmailAndPassword, signOut } from "firebase/auth";
import ReactQuill from "react-quill";
import "react-quill/dist/quill.snow.css";
import "./AdminDashboard.css";

const AdminDashboard = () => {
  const [activeSection, setActiveSection] = useState("registerStudent");
  const [student, setStudent] = useState({
    name: "",
    email: "",
    password: "",
    course: "",
  });
  const [tutor, setTutor] = useState({
    name: "",
    email: "",
    password: "",
    module: "",
  });
  const [course, setCourse] = useState({ name: "" });
  const [modules, setModules] = useState({ courseName: "", moduleName: "" });
  const [newModule, setNewModule] = useState("");
  const [courses, setCourses] = useState([]);
  const [allModules, setAllModules] = useState([]);
  const [users, setUsers] = useState([]);
  const [moduleContent, setModuleContent] = useState({
    moduleName: "",
    textContent: "",
    images: [],
    videos: [],
  });

  const navigate = useNavigate();

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const coursesSnapshot = await getDocs(collection(db, "courses"));
        const coursesList = coursesSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setCourses(coursesList);
      } catch (error) {
        console.error("Error fetching courses: ", error);
      }
    };

    const fetchModules = async () => {
      try {
        const modulesSnapshot = await getDocs(collection(db, "modules"));
        const modulesList = modulesSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setAllModules(modulesList);
      } catch (error) {
        console.error("Error fetching modules: ", error);
      }
    };

    const fetchUsers = async () => {
      try {
        const usersSnapshot = await getDocs(collection(db, "users"));
        const usersList = usersSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setUsers(usersList);
      } catch (error) {
        console.error("Error fetching users: ", error);
      }
    };

    fetchCourses();
    fetchModules();
    fetchUsers();
  }, []);

  const registerStudent = async () => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        student.email,
        student.password
      );
      const user = userCredential.user;

      await setDoc(doc(db, "users", user.uid), {
        name: student.name,
        email: student.email,
        course: student.course,
        role: "student",
      });
      alert("Student registered successfully");
    } catch (e) {
      console.error("Error adding document: ", e);
      alert("Error adding document: " + e.message);
    }
  };

  const registerTutor = async () => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        tutor.email,
        tutor.password
      );
      const user = userCredential.user;

      await setDoc(doc(db, "users", user.uid), {
        name: tutor.name,
        email: tutor.email,
        module: tutor.module,
        role: "tutor",
      });
      alert("Tutor registered successfully");
    } catch (e) {
      console.error("Error adding document: ", e);
      alert("Error adding document: " + e.message);
    }
  };

  const addCourse = async () => {
    try {
      await addDoc(collection(db, "courses"), {
        name: course.name,
        modules: [],
      });
      alert("Course added successfully");
    } catch (e) {
      console.error("Error adding document: ", e);
      alert("Error adding document: " + e.message);
    }
  };

  const assignModuleToCourse = async () => {
    try {
      const courseRef = query(
        collection(db, "courses"),
        where("name", "==", modules.courseName)
      );
      const courseSnapshot = await getDocs(courseRef);
      if (!courseSnapshot.empty) {
        const courseDoc = courseSnapshot.docs[0];
        await updateDoc(doc(db, "courses", courseDoc.id), {
          modules: arrayUnion(modules.moduleName),
        });
        alert("Module assigned successfully");
      } else {
        alert("Course not found");
      }
    } catch (e) {
      console.error("Error assigning module: ", e);
      alert("Error assigning module: " + e.message);
    }
  };

  const addModule = async () => {
    try {
      await addDoc(collection(db, "modules"), {
        name: newModule,
      });
      setNewModule(""); // Clear the input field after adding
      alert("Module added successfully");
    } catch (e) {
      console.error("Error adding module: ", e);
      alert("Error adding module: " + e.message);
    }
  };

  const addModuleContent = async () => {
    try {
      const moduleRef = query(
        collection(db, "modules"),
        where("name", "==", moduleContent.moduleName)
      );
      const moduleSnapshot = await getDocs(moduleRef);
      if (moduleSnapshot.empty) {
        alert("Module not found");
        return;
      }
      const moduleDoc = moduleSnapshot.docs[0];
      await addDoc(collection(db, `modules/${moduleDoc.id}/content`), {
        textContent: moduleContent.textContent,
        images: moduleContent.images,
        videos: moduleContent.videos,
      });
      alert("Module content added successfully");
    } catch (e) {
      console.error("Error adding document: ", e);
      alert("Error adding document: " + e.message);
    }
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    setModuleContent({
      ...moduleContent,
      images: files.map((file) => URL.createObjectURL(file)),
    });
  };

  const handleVideoUpload = (e) => {
    const files = Array.from(e.target.files);
    setModuleContent({
      ...moduleContent,
      videos: files.map((file) => URL.createObjectURL(file)),
    });
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate("/login");
    } catch (e) {
      console.error("Error logging out: ", e);
    }
  };

  const renderSection = () => {
    switch (activeSection) {
      case "registerStudent":
        return (
          <section className="admin-section">
            <h2>Register Student</h2>
            <input
              type="text"
              placeholder="Name"
              onChange={(e) => setStudent({ ...student, name: e.target.value })}
            />
            <input
              type="email"
              placeholder="Email"
              onChange={(e) =>
                setStudent({ ...student, email: e.target.value })
              }
            />
            <input
              type="password"
              placeholder="Password"
              onChange={(e) =>
                setStudent({ ...student, password: e.target.value })
              }
            />
            <select
              onChange={(e) =>
                setStudent({ ...student, course: e.target.value })
              }
            >
              <option value="">Select Course</option>
              {courses.map((course) => (
                <option key={course.id} value={course.name}>
                  {course.name}
                </option>
              ))}
            </select>
            <button className="admin-button-student" onClick={registerStudent}>
              Register Student
            </button>
          </section>
        );
      case "registerTutor":
        return (
          <section className="admin-section">
            <h2>Register Tutor</h2>
            <input
              type="text"
              placeholder="Name"
              onChange={(e) => setTutor({ ...tutor, name: e.target.value })}
            />
            <input
              type="email"
              placeholder="Email"
              onChange={(e) => setTutor({ ...tutor, email: e.target.value })}
            />
            <input
              type="password"
              placeholder="Password"
              onChange={(e) => setTutor({ ...tutor, password: e.target.value })}
            />
            <select
              onChange={(e) => setTutor({ ...tutor, module: e.target.value })}
            >
              <option value="">Select Module</option>
              {allModules.map((module) => (
                <option key={module.id} value={module.name}>
                  {module.name}
                </option>
              ))}
            </select>
            <button className="admin-button-tutor" onClick={registerTutor}>
              Register Tutor
            </button>
          </section>
        );
      case "addCourse":
        return (
          <section className="admin-section">
            <h2>Add Course</h2>
            <input
              type="text"
              placeholder="Course Name"
              onChange={(e) => setCourse({ ...course, name: e.target.value })}
            />
            <button className="admin-button-course" onClick={addCourse}>
              Add Course
            </button>
          </section>
        );
      case "assignModules":
        return (
          <section className="admin-section">
            <h2>Assign Modules to Courses</h2>
            <select
              onChange={(e) =>
                setModules({ ...modules, courseName: e.target.value })
              }
            >
              <option value="">Select Course</option>
              {courses.map((course) => (
                <option key={course.id} value={course.name}>
                  {course.name}
                </option>
              ))}
            </select>
            <select
              onChange={(e) =>
                setModules({ ...modules, moduleName: e.target.value })
              }
            >
              <option value="">Select Module</option>
              {allModules.map((module) => (
                <option key={module.id} value={module.name}>
                  {module.name}
                </option>
              ))}
            </select>
            <button
              className="admin-button-module"
              onClick={assignModuleToCourse}
            >
              Assign Module
            </button>
          </section>
        );
      case "moduleContent":
        return (
          <section className="admin-section">
            <h2>Add Module Content</h2>
            <select
              onChange={(e) =>
                setModuleContent({
                  ...moduleContent,
                  moduleName: e.target.value,
                })
              }
            >
              <option value="">Select Module</option>
              {allModules.map((module) => (
                <option key={module.id} value={module.name}>
                  {module.name}
                </option>
              ))}
            </select>
            <ReactQuill
              value={moduleContent.textContent}
              onChange={(value) =>
                setModuleContent({ ...moduleContent, textContent: value })
              }
              modules={{
                toolbar: [
                  [{ header: "1" }, { header: "2" }, { font: [] }],
                  [{ size: [] }],
                  ["bold", "italic", "underline", "strike", "blockquote"],
                  [
                    { list: "ordered" },
                    { list: "bullet" },
                    { indent: "-1" },
                    { indent: "+1" },
                  ],
                  ["link", "image", "video"],
                  ["clean"],
                ],
              }}
              formats={[
                "header",
                "font",
                "size",
                "bold",
                "italic",
                "underline",
                "strike",
                "blockquote",
                "list",
                "bullet",
                "indent",
                "link",
                "image",
                "video",
              ]}
            />
            <input type="file" multiple onChange={handleImageUpload} />
            <input
              type="file"
              accept="video/*"
              multiple
              onChange={handleVideoUpload}
            />
            <button className="admin-button-content" onClick={addModuleContent}>
              Add Content
            </button>
          </section>
        );
      case "allUsers":
        return (
          <section className="admin-section">
            <h2>All Users</h2>
            <table className="users-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Role</th>
                  <th>Email</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id}>
                    <td>{user.name}</td>
                    <td>{user.role}</td>
                    <td>{user.email}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        );
      case "addModule":
        return (
          <section className="admin-section">
            <h2>Add Module</h2>
            <input
              type="text"
              placeholder="Module Name"
              value={newModule}
              onChange={(e) => setNewModule(e.target.value)}
            />
            <button className="admin-button-module" onClick={addModule}>
              Add Module
            </button>
          </section>
        );
      default:
        return null;
    }
  };

  return (
    <div className="admin-dashboard">
      <div className="admin-sidebar">
        <div>
          <h1>Admin</h1>
          <nav>
            <ul>
              <li
                className={activeSection === "registerStudent" ? "active" : ""}
                onClick={() => setActiveSection("registerStudent")}
              >
                Register Student
              </li>
              <li
                className={activeSection === "registerTutor" ? "active" : ""}
                onClick={() => setActiveSection("registerTutor")}
              >
                Register Tutor
              </li>
              <li
                className={activeSection === "addCourse" ? "active" : ""}
                onClick={() => setActiveSection("addCourse")}
              >
                Add Course
              </li>
              <li
                className={activeSection === "assignModules" ? "active" : ""}
                onClick={() => setActiveSection("assignModules")}
              >
                Assign Modules
              </li>
              <li
                className={activeSection === "moduleContent" ? "active" : ""}
                onClick={() => setActiveSection("moduleContent")}
              >
                Module Content
              </li>
              <li
                className={activeSection === "allUsers" ? "active" : ""}
                onClick={() => setActiveSection("allUsers")}
              >
                All Users
              </li>
              <li
                className={activeSection === "addModule" ? "active" : ""}
                onClick={() => setActiveSection("addModule")}
              >
                Add Module
              </li>
            </ul>
          </nav>
        </div>
        <button className="admin-logout-button" onClick={handleLogout}>
          Logout
        </button>
      </div>
      <div className="admin-main-content">{renderSection()}</div>
    </div>
  );
};

export default AdminDashboard;
