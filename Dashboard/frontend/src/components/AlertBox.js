import React from "react";
import "../styles.css";

const AlertBox = ({ message, onClose, title = "Success" }) => {
  return (
    <div className="modal-backdrop">
      <div className="modal-box">
        <h2 className="modal-title">{title}</h2>
        <p className="message-text">{message}</p>
        <div className="modal-actions">
          <button onClick={onClose} className="submit-button">
            OK
          </button>
        </div>
      </div>
    </div>
  );
};

export default AlertBox;
