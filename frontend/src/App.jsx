import { BrowserRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import LoginPage from "./pages/Login";
import Layout from "./pages/Layout"
import HomePage from "./pages/Homepage";
import Machines from "./pages/Machines";
import Tickets from "./pages/Tickets";
import Technicians from "./pages/Technicians";
import Monitoring from "./pages/Monitoring";
import Analytics from "./pages/Analytics";
import Profile from "./pages/Profile";
import Alerts from "./pages/Alerts"

// const AdminOnly = ({ role, children }) => {
//   return role === "administrator" ? children : <Navigate to="/homepage" />;
//   };

// const role = new URLSearchParams(window.location.search).get("user");

// Protection for ANY logged-in user
const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem("token");
  return token ? children : <Navigate to="/" />;
};

// Protection specifically for Admins
const AdminOnly = ({ children }) => {
  const role = localStorage.getItem("userRole");
  return role === "ADMIN" ? children : <Navigate to="/homepage" />;
};

function App() {
  return (
    <Router>
      <Routes>
        {/* Public Route */}
        <Route path="/" element={<LoginPage />} />

        {/* Protected Routes (Require Login) */}
        <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
          <Route path="/homepage" element={<HomePage />} />
          <Route path="/machines" element={<Machines />} />
          <Route path="/tickets" element={<Tickets />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/profile" element={<Profile />} />

          {/* Admin Specific Routes */}
          <Route path="/analytics" element={<AdminOnly><Analytics /></AdminOnly>} />
          <Route path="/alerts" element={<AdminOnly><Alerts /></AdminOnly>} />
          <Route path="/technicians" element={<AdminOnly><Technicians /></AdminOnly>} />
        </Route>

        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}

export default App;