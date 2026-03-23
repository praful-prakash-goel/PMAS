import { useState } from "react";
import { useNavigate } from "react-router-dom";
import loginStyles from "../styles/login.module.css";
import loginLogo from "../assets/senseact_logo.png";

const LoginPage = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("");

  const handleLogin = async () => {
    if (!email || !password || !role) {
      alert("Please fill in all fields.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, user_type: role }),
      });

      const data = await response.json();
      console.log(data);

      if (response.ok) {
        alert("Login successful!");
        const userId = data.user_id;
        navigate(`/homepage?user=${role}&user_id=${userId}`);
      } else {
        alert(data.message || "Login failed.");
      }
    } catch (err) {
      console.error("Login error:", err);
      alert("Server error. Please try again later.");
    }
  };

  return (
    <main className={loginStyles.page}>
      <img src={loginLogo} alt="Senseact Logo" className={loginStyles.loginLogo} />

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

        <div className={loginStyles.inputGroup}>
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
        </div>

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