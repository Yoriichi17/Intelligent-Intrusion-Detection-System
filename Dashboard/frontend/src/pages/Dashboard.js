import React, { useEffect, useState } from "react";
import Header from "../components/Header";
import AddUserModal from "../components/AddUserModal";
import AlertBox from "../components/AlertBox";
import { getPredictions , retrainModel } from "../api";
import { getToken, removeToken } from "../utils";
import { useNavigate } from "react-router-dom";
import { Pie } from "react-chartjs-2";
import html2pdf from "html2pdf.js";
import "../styles.css";
// import "../css/dashboard.css";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const Dashboard = () => {
  const [threats, setThreats] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [showMessageBox, setShowMessageBox] = useState(false);

  const [page, setPage] = useState(1);
  const [allThreats, setAllThreats] = useState([]);

  const pageSize = 200;
  const navigate = useNavigate();

  const fetchData = async (pageNum) => {
    const token = getToken();
    if (!token) return navigate("/");
    try {
      const res = await getPredictions(token, pageNum, pageSize);
      setThreats(res.data);
    } catch {
      alert("Failed to fetch threats.");
    }
  };

  const fetchAllData = async () => {
    const token = getToken();
    if (!token) return navigate("/");
    try {
      let allData = [];
      let currentPage = 1;
      let hasMore = true;

      while (hasMore) {
        const res = await getPredictions(token, currentPage, pageSize);
        const data = res.data;
        allData = [...allData, ...data];
        if (data.length < pageSize) {
          hasMore = false;
        } else {
          currentPage += 1;
        }
      }

      setAllThreats(allData);
    } catch {
      alert("Failed to fetch all threats.");
    }
  };

  useEffect(() => {
    fetchData(page);
    fetchAllData();

    const interval = setInterval(() => {
      fetchData(page);
      fetchAllData();
    }, 1000);

    return () => clearInterval(interval);
  }, [page]);

  const pieData = {
    labels: [...new Set(allThreats.map((t) => t.prediction))],
    datasets: [
      {
        data: Object.values(
          allThreats.reduce((acc, t) => { 
            acc[t.prediction] = (acc[t.prediction] || 0) + 1;
            return acc;
          }, {})
        ),
        backgroundColor: ["#ff6384", "#36a2eb", "#ffcd56", "#4bc0c0"],
      },
    ],
  };

  const downloadPDF = () => {
    const element = document.getElementById("dashboard-content");
    html2pdf().from(element).save("threat-report.pdf");
  };

  const handleRetrain = async () => {
  const token = getToken();
  if (!token) return navigate("/");
  try {
    await retrainModel(token);
    setShowMessageBox(true);
  } catch {
    alert("Model retraining failed.");
  }
};


  const logout = () => {
    removeToken();
    navigate("/");
  };
  return (
    <div className="dashboard-container">
      <Header onAddUser={() => setShowModal(true)} onLogout={logout} handleRetrain={handleRetrain}/>
      {showModal && <AddUserModal onClose={() => setShowModal(false)} />}
      {showMessageBox && (
      <AlertBox
        title="Model Retrained"
        message="The model was retrained successfully."
        onClose={() => setShowMessageBox(false)}
      />
      )}
      <div
        className={showModal ? "blurred" : "dashboard-content"}
        id="dashboard-content"
      >
        <div className="dashboard-data-container">
          <div className="report-header">
            <h2>Threat Report</h2>
            <button onClick={downloadPDF} className="btn pdf-button">
              Download PDF
            </button>
          </div>
          <div id="report">
            <table className="threat-table" id="threat-table">
              <thead>
                <tr>
                  <th className="ts">Timestamp</th>
                  <th className="sip">Source IP</th>
                  <th className="dip">Destination IP</th>
                  <th className="p">Type</th>
                </tr>
              </thead>
              <tbody>
                {threats.map((t, i) => (
                  <tr key={i} className="threat-row"
                  style={
                    ["DOS", "SQLI", "BruteForce", "BRUTEFORCE"].includes(t.prediction)
                      ? { backgroundColor: "#8b0000" }
                      : {}
                  }
                  >
                    <td className="tstd">{t.timestamp}</td>
                    <td className="siptd">{t.source_ip}</td>
                    <td className="td">{t.destination_ip}</td>
                    <td className="ptd">{t.prediction}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination-controls">
            <button
              onClick={() => setPage((prev) => Math.max(prev - 1, 1))}
              disabled={page === 1}
              className="btn nav-button"
            >
              ← Prev
            </button>
            <span className="page-number">Page {page}</span>
            <button
              onClick={() => setPage((prev) => prev + 1)}
              disabled={threats.length < pageSize}
              className="btn nav-button"
            >
              Next →
            </button>
          </div>
        </div>

        <div className="chart-wrapper" id="download-content">
          <Pie data={pieData} className="pie" />
          <div className="chart-legend">
            <h3>Traffic Distribution</h3>
            <div className="legend-list">
              {pieData.labels.map((label, index) => (
                <div
                  className="data-card-threat"
                  key={index}
                  style={{ color: pieData.datasets[0].backgroundColor[index] }}
                >
                  <div>
                    <b>{label}</b>
                  </div>
                  <div>
                    <b>{pieData.datasets[0].data[index]}</b>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
