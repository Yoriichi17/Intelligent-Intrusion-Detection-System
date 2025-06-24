import React, { useState } from "react";
import { register } from "../api";
import "../styles.css";

const AddUserModal = ({ onClose }) => {
  const [form, setForm] = useState({ username: "", password: "" });
  const [message, setMessage] = useState("");

  const handleSubmit = async () => {
    try {
      await register(form);
      setMessage("User added successfully.");
      setTimeout(() => onClose(), 1000);
    } catch {
      setMessage("Failed to add user.");
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-box">
        <h2 className="modal-title">Add New User</h2>
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
        <div className="modal-actions">
          <button onClick={onClose} className="cancel-button">
            Cancel
          </button>
          <button onClick={handleSubmit} className="submit-button">
            Submit
          </button>
        </div>
        {message && <p className="message-text">{message}</p>}
      </div>
    </div>
  );
};

export default AddUserModal;
