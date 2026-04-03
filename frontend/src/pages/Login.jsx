import { useState } from "react";
import { useNavigate } from "react-router-dom";
import loginStyles from "../styles/login.module.css";

const LoginPage = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    // Removed 'role' check since we're fetching it from backend
    if (!email || !password) {
      alert("Please fill in all fields.");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }), // Sending only email and password
      });

      const data = await response.json();

      if (response.ok) {
        alert("Login successful!");
        
        // Extract data from the "user" object returned by your Python code
        // const userRole = data.user.role; 
        
        // instead of passing user_role as url parameters, user_role will be
        // extracted from token stored in local storage for security
        const token = data.access_token;
        //console.log(token)

        // Store token for future authenticated requests
        localStorage.setItem("token", token);
        localStorage.setItem("userRole", data.user.role);
        localStorage.setItem("userId", data.user.id);
        console.log(data.user.name);
        localStorage.setItem("userName", data.user.name);
        localStorage.setItem("userOrg", data.user.org_name);
        localStorage.setItem("userEmail", data.user.email);

        // Navigate using the role fetched from the database
        navigate(`/homepage`);
      } else {
        // FastAPI returns errors in a 'detail' field
        alert(data.detail || "Login failed.");
      }
    } catch (err) {
      console.error("Login error:", err);
      alert("Server error. Please try again later.");
    }
  };

  return (
    <main className={loginStyles.page}>
      <div className={loginStyles.logo} onClick={() => navigate(`/homepage${location.search}`)}>
        SENSE<span>ACT</span>
      </div>
      <div className={loginStyles.loginBox}>
        <h2 className={loginStyles.title}>Welcome Back</h2>

        <div className={loginStyles.inputGroup}>
          <label>Organisation Email*</label>
          <input
            type="email"
            placeholder="Enter your email"
            className={loginStyles.inputField}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        <div className={loginStyles.inputGroup}>
          <label>Password*</label>
          <input
            type="password"
            placeholder="Enter your password"
            className={loginStyles.inputField}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        {/* <div className={loginStyles.inputGroup}>
          <label>Role*</label>
          <select
            className={loginStyles.inputField}
            value={role}
            onChange={(e) => setRole(e.target.value)}
          >
            <option value="">Select your role</option>
            <option value="administrator">Administrator</option>
            <option value="technician">Technician</option>
          </select>
        </div> */}

        <button className={loginStyles.continueBtn} onClick={handleLogin}>
          Continue
        </button>

        {/* <p className={loginStyles.signupText}>
          Don't have an account?{" "}
          <span onClick={() => navigate("/signup")} className={loginStyles.signupLink}>
            Sign Up
          </span>
        </p> */}
      </div>
    </main>
  );
};

export default LoginPage;