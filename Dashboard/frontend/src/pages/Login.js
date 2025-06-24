import React, { useState } from "react";
import { login } from "../api";
import { saveToken } from "../utils";
import { useNavigate } from "react-router-dom";
import "../styles.css";

const Login = () => {
  const [form, setForm] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const res = await login(form);
      saveToken(res.data.access_token);
      navigate("/dashboard");
    } catch (error) {
      setError("Login failed. Check credentials.");
      console.error("Login error:", error);
    }
  };

  return (
    <div className="login-container">
      <h1>Intelligent IDS</h1>
      {/* <h1>CyberThreat Intelligence</h1> */}
      <div className="login-box">
        <h2 className="login-title">Login</h2>
        <input
          className="input-field"
          placeholder="Username"
          value={form.username}
          onChange={(e) => setForm({ ...form, username: e.target.value })}
        />
        <input
          className="input-field"
          placeholder="Password"
          type="password"
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
        />
        {error && <p className="error-text">{error}</p>}
        <button onClick={handleLogin} className="login-button">
          Login
        </button>
      </div>
      {/* new user? Register */}
    </div>
  );
};

export default Login;
