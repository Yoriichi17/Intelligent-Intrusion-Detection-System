import React from "react";
import "../styles.css";

const Header = ({ onAddUser, onLogout, handleRetrain }) => (
  <div className="header-bar">
    {/* <h1 className="header-title">CyberThreat Intelligence</h1> */}
    <h1 className="header-title">Intelligent IDS</h1>
    <div>
      <button onClick={handleRetrain} className="btn add-user-button">
        Update Model
      </button>
      <button onClick={onAddUser} className="btn add-user-button">
        Add User
      </button>
      <button onClick={onLogout} className="btn logout-button">
        Logout
      </button>
    </div>
  </div>
);

export default Header;
