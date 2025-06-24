import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

export const register = (user) => axios.post(`${API_BASE}/register`, user);

export const login = (user) => axios.post(`${API_BASE}/login`, user);


export const getPredictions = (token, page = 1, pageSize = 500) =>
  axios.get(`${API_BASE}/predictions?page=${page}&page_size=${pageSize}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

export const retrainModel = (token) =>
  axios.post(`${API_BASE}/manual_retrain`, {}, {
    headers: { Authorization: `Bearer ${token}` },
  });
